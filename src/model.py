from math import floor, ceil, prod
from typing import cast, Mapping, Any

from attrs import frozen, asdict

# import matplotlib.pyplot as plt
# import seaborn as sns
import pytorch_lightning as pl
import torch
import torch.autograd
from einops import rearrange, reduce
from torch import optim, nn, Tensor
from torch.nn.utils import weight_norm  # type: ignore[attr-defined]
from torch.distributions.normal import Normal

from .loss import MultiScaleSTFTLoss, feature_matching_loss, hinge_gan_losses
# from .plots import draw_histogram
from .noise import Noise, NoiseConfig
from .discriminators import Discriminator  # type: ignore[attr-defined]
from .activations import ACTIVATIONS
from .pqmf import PQMF
from .normalizing_flows import RealNVP

SAMPLING_RATE = 48000


def _get_padding(kernel_size: int, dilation: int) -> int:
    """
    returns "centered" padding that preserves the input length
    """

    # for simplicity, kernel sizes are usually odd anyway - 3, 5, 7
    assert kernel_size % 2 == 1

    p = dilation * (kernel_size - 1) // 2
    return p


def _get_strided_padding(stride: int) -> tuple[int, int]:
    """
    returns left/right padding for the downsampling strided convolutions, made
    to have the downsampling ratio be exactly `1/stride`

    assumes kernel size to be `2*stride`
    """

    half = stride / 2
    return floor(half), ceil(half)


def _instantiate_activation(name: str, channels: int) -> nn.Module:
    return ACTIVATIONS[name](channels)


class ZeroPad1d(nn.Module):
    def __init__(self, padding_left: int, padding_right: int) -> None:
        super().__init__()
        self._left = padding_left
        self._right = padding_right

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.pad(x, (self._left, self._right))


@frozen
class ModelConfig:
    capacity: int
    latent_size: int
    dilations: list[list[int]]
    strides: list[int]
    latent_loss_weight: float = 0.5
    reconstruction_loss_weight: float = 1.0
    adversarial_loss_weight: float = 1.0
    feature_loss_weight: float = 1.0
    adam_betas: tuple[float, float] = (0.5, 0.9)
    monitor_grad_norm: bool = False
    detect_nans: bool = False
    discriminator_lr: float = 1e-4
    initial_lr: float = 1e-4
    final_lr: float = 1e-5
    do_amp_mod: bool = True
    do_weight_norm: bool = True
    activation: str = "relu"
    n_bands: int = 16
    fixed_length: int | None = None
    mono: bool = False
    nf_posterior_layers: int | None = None
    nf_prior_layers: int | None = None
    prior_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        assert len(self.dilations) == len(
            self.strides
        ), "strides and dilations need to be the same length (which is the desired amount of encoder/decoder blocks)"

        assert (self.nf_posterior_layers is None) or (self.fixed_length is not None), "must be training with single latent vector if using normalizing flows"
        assert (self.nf_prior_layers is None) or (self.fixed_length is not None), "must be training with single latent vector if using normalizing flows"


class _ResidualDilatedUnit(nn.Module):
    kernel_size = 7

    def __init__(
        self,
        channels: int,
        dilation: int,
        do_weight_norm: bool,
        activation: str,
    ):
        super().__init__()

        def _weightnorm(m: nn.Module) -> nn.Module:
            return m if not do_weight_norm else weight_norm(m)

        self.net = nn.Sequential(
            _instantiate_activation(activation, channels),
            _weightnorm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    padding=_get_padding(self.kernel_size, dilation),
                )
            ),
            _instantiate_activation(activation, channels),
            _weightnorm(nn.Conv1d(channels, channels, kernel_size=1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# ================================== ENCODER =============================================


class _EncoderBlock(nn.Module):
    """
    the downsampling factor of this encoder block is
    ```
    L_out = floor(L_in / stride)
    ```
    """

    def __init__(
        self,
        in_channels: int,
        dilations: list[int],
        stride: int,
        do_weight_norm: bool,
        activation: str,
    ):
        super().__init__()
        # assert stride % 2 == 0, f"stride should be even, got {stride}"

        def _weightnorm(m: nn.Module) -> nn.Module:
            return m if not do_weight_norm else weight_norm(m)

        self.net = nn.Sequential(
            *[
                _ResidualDilatedUnit(in_channels, dilation=d, do_weight_norm=do_weight_norm, activation=activation)
                for d in dilations
            ],
            _instantiate_activation(activation, in_channels),
            # asymmetrical padding might be breaking something, try sym
            ZeroPad1d(*_get_strided_padding(stride)),
            _weightnorm(
                nn.Conv1d(
                    in_channels,
                    2 * in_channels,
                    kernel_size=2 * stride,
                    padding=0,
                    # padding=stride // 2,
                    stride=stride,
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(nn.Module):
    input_kernel_size = 7
    output_kernel_size = 3

    def __init__(
        self,
        start_channels: int,
        dilations: list[list[int]],
        strides: list[int],
        latent_size: int,
        do_weight_norm: bool,
        activation: str,
        n_bands: int,
        audio_channels: int,
        fixed_length: int | None = None,
    ):
        super().__init__()
        assert len(dilations) == len(strides)
        self.latent_size = latent_size

        encoder_blocks = [
            _EncoderBlock(start_channels * 2**i, block_dilations, stride, do_weight_norm, activation=activation)
            for i, (block_dilations, stride) in enumerate(zip(dilations, strides))
        ]

        def _weightnorm(m: nn.Module) -> nn.Module:
            return m if not do_weight_norm else weight_norm(m)

        if fixed_length is not None:
            assert round(fixed_length / prod(strides)) == fixed_length / prod(strides)
        single_latent_reduction: list[nn.Module] = (
            []
            if fixed_length is None
            else [_weightnorm(nn.Linear(fixed_length // prod(strides), 1))]
        )

        self.net = nn.Sequential(
            _weightnorm(
                nn.Conv1d(
                    n_bands * audio_channels,
                    start_channels,
                    kernel_size=self.input_kernel_size,
                    padding=_get_padding(self.input_kernel_size, 1),
                )
            ),
            *encoder_blocks,
            _instantiate_activation(activation, start_channels * 2 ** len(strides)),
            _weightnorm(
                nn.Conv1d(
                    start_channels * 2 ** len(strides),
                    latent_size * 2,  # outputs mean and log-variance for each dim
                    kernel_size=self.output_kernel_size,
                    padding=_get_padding(self.output_kernel_size, 1),
                )
            ),
            *single_latent_reduction,
        )

        self.register_buffer("freeze_weights", torch.tensor(0))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.net(x)
        mean = encoded[:, : self.latent_size, :]
        logvar = encoded[:, self.latent_size :, :]

        if self.freeze_weights:  # type: ignore[has-type]
            mean = mean.detach()
            logvar = logvar.detach()

        return mean, logvar

    def freeze(self, value: bool) -> None:
        self.freeze_weights = torch.tensor(int(value), device=self.freeze_weights.device)  # type: ignore[has-type]


# ================================== DECODER =============================================


class _DecoderBlock(nn.Module):
    """
    the upsampling factor of this encoder block is
    ```
    L_out = L_in * stride
    ```
    """

    def __init__(
        self,
        in_channels: int,
        dilations: list[int],
        stride: int,
        do_weight_norm: bool,
        activation: str,
    ):
        super().__init__()
        # again, for simplicity - this will always be the case
        assert in_channels % 2 == 0

        def _weightnorm(m: nn.Module) -> nn.Module:
            return m if not do_weight_norm else weight_norm(m)

        self.net = nn.Sequential(
            _instantiate_activation(activation, in_channels),
            _weightnorm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=2 * stride,
                    padding=_get_strided_padding(stride)[1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            ),
            *[_ResidualDilatedUnit(in_channels // 2, dilation=d, do_weight_norm=do_weight_norm, activation=activation) for d in dilations],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    input_kernel_size = 3
    output_kernel_size = 7

    def __init__(
        self,
        start_channels: int,
        dilations: list[list[int]],
        strides: list[int],
        latent_size: int,
        noise_config: NoiseConfig | None,
        do_amp_mod: bool,
        do_weight_norm: bool,
        activation: str,
        n_bands: int,
        audio_channels: int,
        fixed_length: int | None = None,
    ):
        super().__init__()
        assert len(dilations) == len(strides)
        self.latent_size = latent_size
        self.do_amp_mod = do_amp_mod

        decoder_blocks = [
            _DecoderBlock(start_channels // 2**i, block_dilations, stride, do_weight_norm, activation=activation)
            for i, (block_dilations, stride) in enumerate(zip(dilations, strides))
        ]

        def _weightnorm(m: nn.Module) -> nn.Module:
            return m if not do_weight_norm else weight_norm(m)

        if fixed_length is not None:
            assert round(fixed_length / prod(strides)) == fixed_length / prod(strides)
        single_latent_upsampling: list[nn.Module] = (
            []
            if fixed_length is None
            else [_weightnorm(nn.Linear(1, fixed_length // prod(strides)))]
        )

        self.net = nn.Sequential(
            *single_latent_upsampling,
            _weightnorm(
                nn.Conv1d(
                    latent_size,
                    start_channels,
                    kernel_size=self.input_kernel_size,
                    padding=_get_padding(self.input_kernel_size, 1),
                )
            ),
            *decoder_blocks,
            _instantiate_activation(activation, start_channels // 2 ** len(strides)),
        )

        amp_out_channels = 2 if do_amp_mod else 1
        self.waveform_amp_net = _weightnorm(
            nn.Conv1d(
                start_channels // 2 ** len(strides),
                n_bands * audio_channels * amp_out_channels,
                kernel_size=self.output_kernel_size,
                padding=_get_padding(self.output_kernel_size, 1),
            )
        )

        self.noise_net = (
            Noise(
                in_channels=start_channels // 2 ** len(strides),
                n_bands=n_bands,
                audio_channels=audio_channels,
                **asdict(noise_config),
            )
            if noise_config is not None
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        x_hat = self.net(x)

        waveform_amp = self.waveform_amp_net(x_hat)
        if self.do_amp_mod:
            waveform, amp_mod = torch.split(
                waveform_amp,
                waveform_amp.shape[1] // 2,
                dim=1,
            )
            amp_modded = waveform * torch.sigmoid(amp_mod)
        else:
            amp_modded = waveform_amp  # not actually amp moded, mind you

        if self.noise_net is not None:
            noise = self.noise_net(x_hat)
            return torch.tanh(amp_modded + noise)
        else:
            return amp_modded


class VAE(pl.LightningModule):
    def __init__(
        self,
        capacity: int,
        latent_size: int,
        dilations: list[list[int]],
        strides: list[int],
        noise_config: NoiseConfig | None,
        stft_window_sizes: list[int] | None = None,
        latent_loss_weight: float = 0.5,
        reconstruction_loss_weight: float = 1.0,
        adversarial_loss_weight: float = 1.0,
        feature_loss_weight: float = 1.0,
        adam_betas: tuple[float, float] = (0.5, 0.9),
        lr_decay_steps: int = 500000,
        monitor_grad_norm: bool = False,
        detect_nans: bool = False,
        discriminator_lr: float = 1e-4,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
        do_amp_mod: bool = True,
        do_weight_norm: bool = True,
        activation: str = "relu",
        n_bands: int = 16,
        fixed_length: int | None = None,
        mono: bool = False,
        nf_posterior_layers: int | None = None,
        nf_prior_layers: int | None = None,
        prior_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert len(dilations) == len(strides)

        self.audio_channels = 1 if mono else 2

        self.encoder = Encoder(
            start_channels=capacity,
            dilations=dilations,
            strides=strides,
            latent_size=latent_size,
            do_weight_norm=do_weight_norm,
            activation=activation,
            n_bands=n_bands,
            audio_channels=self.audio_channels,
            fixed_length=fixed_length,
        )
        self.posterior = RealNVP(num_layers=nf_posterior_layers) if nf_posterior_layers is not None else None
        self.prior = RealNVP(num_layers=nf_prior_layers) if nf_prior_layers is not None else None
        self.decoder = Decoder(
            start_channels=capacity * 2 ** len(strides),
            dilations=dilations,
            strides=strides[::-1],
            latent_size=latent_size,
            noise_config=noise_config,
            do_amp_mod=do_amp_mod,
            do_weight_norm=do_weight_norm,
            activation=activation,
            n_bands=n_bands,
            audio_channels=self.audio_channels,
            fixed_length=fixed_length,
        )
        self.pqmf = PQMF(100, n_bands, n_channels=self.audio_channels)

        self.discriminator = Discriminator(n_channels=self.audio_channels, sample_rate=SAMPLING_RATE)

        _window_sizes = stft_window_sizes or [2048, 1024, 512, 256, 128]
        self.reconstruction_loss = MultiScaleSTFTLoss(_window_sizes)

        self.latent_loss_weight = latent_loss_weight
        self.prior_loss_weight = prior_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.betas = adam_betas
        self.lr_decay_steps = lr_decay_steps
        self.monitor_grad_norm = monitor_grad_norm
        self.detect_nans = detect_nans
        self.discriminator_lr = discriminator_lr
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.validation_outputs: dict[str, list[Tensor]] = {
            "audio": [],
            "latent": [],
            "original": [],
        }
        self.validation_epoch = 0
        self.discrimination_phase = False
        self.discriminator_warmup_phase = False

        self.automatic_optimization = False  # PL doesn't support discriminator learning out of the box

    def _reparametrize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)  # type: ignore
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def _split_bands(self, x: Tensor) -> Tensor:
        x_rearranged = cast(Tensor, rearrange(x, "b (c one) t -> (b c) one t", one=1))
        multiband_x = self.pqmf(x_rearranged)
        return cast(Tensor, rearrange(multiband_x, "(b chs) bands t -> b (chs bands) t", b=x.shape[0]))

    def _join_bands(self, multiband_x: Tensor) -> Tensor:
        x_rearranged = cast(Tensor, rearrange(multiband_x, "b (chs c) t -> (b chs) c t", chs=self.audio_channels))
        single_band_x = self.pqmf.inverse(x_rearranged)
        return cast(Tensor, rearrange(single_band_x, "(b chs) c t -> b (chs c) t", chs=self.audio_channels))

    @staticmethod
    def _latent_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return (mean.pow(2) + torch.exp(logvar) - logvar - 1).sum(1).mean()

    @staticmethod
    def _nf_latent_loss(z0: Tensor, mean: Tensor, logvar: Tensor, z: Tensor, log_det: Tensor) -> Tensor:
        q0 = Normal(mean, torch.exp(0.5 * logvar))
        iso_gaussian = Normal(0.0, 1.0)
        return (q0.log_prob(z0).sum(-1) - iso_gaussian.log_prob(z).sum(-1) - log_det).mean()  # TODO: shapes?

    def _prior_loss(self, z: Tensor) -> Tensor:
        return self.prior.kld(z, Normal(0.0, 1.0))

    def _discrimination(self, x: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        all_discriminators_real_feature_maps = self.discriminator(x)
        all_discriminators_fake_feature_maps = self.discriminator(x_hat)
        n_discriminators = len(all_discriminators_real_feature_maps)

        disc_losses, gen_losses = zip(*[
            hinge_gan_losses(disc_real[-1], disc_fake[-1])
            for disc_real, disc_fake in zip(all_discriminators_real_feature_maps, all_discriminators_fake_feature_maps)
        ])

        feature_loss = cast(
            Tensor,
            sum(
                [
                    feature_matching_loss(disc_real[1:-1], disc_fake[1:-1])
                    for disc_real, disc_fake in zip(all_discriminators_real_feature_maps, all_discriminators_fake_feature_maps)
                ]
            ) / n_discriminators,
        )
        gen_loss = cast(Tensor, sum(gen_losses))
        disc_loss = cast(Tensor, sum(disc_losses))

        return gen_loss, disc_loss, feature_loss

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        mean, logvar = self.encoder(self._split_bands(x))
        z = self._reparametrize(mean, logvar)

        if self.posterior is not None:
            z_flow, log_det = self.posterior(z)
        else:
            z_flow = z
            log_det = torch.zeros(())  # TODO size of this?

        x_hat = self._join_bands(self.decoder(z_flow))

        losses_dict = {
            "latent_loss": (
                self._latent_loss(mean, logvar)
                if self.posterior is None
                else self._nf_latent_loss(z, mean, logvar, z_flow, log_det)
            ),
            "reconstruction_loss": self.reconstruction_loss(x, x_hat),
        }
        total_loss = (
            self.latent_loss_weight * losses_dict["latent_loss"]
            + self.reconstruction_loss_weight * losses_dict["reconstruction_loss"]
        )

        if self.discrimination_phase:
            adv_loss, disc_loss, feature_loss = self._discrimination(x, x_hat)
            losses_dict.update({
                "adversarial_loss": adv_loss,
                "feature_matching_loss": feature_loss,
                "discriminator_loss": disc_loss,
            })
            total_loss += (
                self.adversarial_loss_weight * losses_dict["adversarial_loss"]
                + self.feature_loss_weight * losses_dict["feature_matching_loss"]
            )

            if self.prior is not None:
                prior_loss = self._prior_loss(z_flow)
                losses_dict["prior_loss"] = prior_loss
                total_loss += self.prior_loss_weight * prior_loss

        return x_hat, z, total_loss, losses_dict

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.encoder.freeze(self.discrimination_phase)
        gen_opt, disc_opt = self.optimizers()  # type: ignore[attr-defined]
        _, _, gen_loss, losses_dict = self.forward(batch)

        self.log("loss/train_loss", gen_loss, on_step=False, on_epoch=True)
        for loss_key, value_tensor in losses_dict.items():
            self.log(f"loss/{loss_key}", value_tensor, on_step=False, on_epoch=True)
        self.log("latent_loss_weight", self.latent_loss_weight, on_step=False, on_epoch=True)

        if self.discriminator_warmup_phase or self._update_discriminator(batch_idx):
            disc_opt.zero_grad()
            losses_dict["discriminator_loss"].backward()
            disc_opt.step()
        else:
            lr_schedule = self.lr_schedulers()
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            lr_schedule.step()

    def _update_discriminator(self, batch_idx: int) -> bool:
        update_discriminator_every = 4
        return (batch_idx % update_discriminator_every == 0) and self.discrimination_phase

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        reconstructed_audio, z, loss, losses_dict = self.forward(batch)
        self.log("loss/validation_loss", loss)
        self.log("loss/validation_reconstruction_loss", losses_dict["reconstruction_loss"])
        self.log("loss/validation_latent_loss", losses_dict["latent_loss"])

        self.validation_outputs["original"].append(batch)
        self.validation_outputs["audio"].append(reconstructed_audio)
        self.validation_outputs["latent"].append(z)

    @staticmethod
    def _mono_concatenate_batch(audio: Tensor) -> Tensor:
        assert len(audio.shape) == 3, f"got unexpected audio shape: {audio.shape}"
        mono_audio: Tensor = reduce(audio, "b c l -> b l", "mean")
        audio_concatenated: Tensor = cast(Tensor, rearrange(mono_audio, "b l -> (b l)")).cpu()
        return audio_concatenated

    def on_validation_epoch_end(self) -> None:
        # TODO: maybe we want more?
        # validation_audio = rearrange(
        #     self.validation_outputs,
        #     "li b c len -> (li b) c len",
        # )
        # taking second batch cuz why not
        validation_audio = self.validation_outputs["audio"][1]
        audio_concatenated = self._mono_concatenate_batch(validation_audio)
        self.logger.experiment.add_audio(  # type: ignore
            "validation_audio",
            audio_concatenated.numpy(),
            self.validation_epoch,
            SAMPLING_RATE,
        )
        if self.validation_epoch <= 0:
            original_audio = self.validation_outputs["original"][1]
            self.logger.experiment.add_audio(  # type: ignore
                "validation_audio_original",
                self._mono_concatenate_batch(original_audio).numpy(),
                self.validation_epoch,
                SAMPLING_RATE,
            )

        # self.logger.experiment.add_histogram(  # type: ignore
        #     "validation_audio_histogram",
        #     audio_concatenated.numpy(),
        #     self.validation_epoch,
        #     bins="fd",
        # )
        # histograms don't work and are expensive and all that, eh
        self.log("audio_stats/validation_audio_min", audio_concatenated.min())
        self.log("audio_stats/validation_audio_max", audio_concatenated.max())
        self.log("audio_stats/validation_audio_mean", audio_concatenated.mean())

        # also expensive it seems, eh
        # self.logger.experiment.add_figure(  # type: ignore
        #     "validation_audio_histogram",
        #     draw_histogram(audio_concatenated.numpy()),
        #     self.validation_epoch,
        # )

        # unnecessary for now
        # validation_embeddings = self.validation_outputs["latent"][0]
        # embeddings_concatenated: Tensor = rearrange(validation_embeddings, "b d l -> (b l) d")
        # assert len(validation_embeddings.shape) == 3, f"got unexpected embedding shape: {validation_embeddings.shape}"
        # self.logger.experiment.add_embedding(  # type: ignore
        #     embeddings_concatenated.cpu().numpy(),
        #     tag="latent space",
        #     global_step=self.validation_epoch,
        # )

        self.validation_outputs["audio"] = []
        self.validation_outputs["latent"] = []
        self.validation_outputs["original"] = []
        self.validation_epoch += 1

        if self.detect_nans:
            torch.autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]

    def configure_optimizers(self) -> optim.Optimizer:
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=self.betas)
        optimizer = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=self.initial_lr, betas=self.betas)
        lr_schedule = optim.lr_scheduler.LinearLR(
            optimizer, 1.0, self.final_lr / self.initial_lr, self.lr_decay_steps
        )
        return (  # type: ignore
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_schedule},
            },
            {
                "optimizer": disc_optimizer,
            }
        )

    def on_train_batch_end(self, x_hat: Tensor | Mapping[str, Any] | None, batch: Tensor, batch_idx: int) -> None:
        if self.monitor_grad_norm:
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in [*self.encoder.parameters(), *self.decoder.parameters()]
                if p.grad is not None
            ) ** (1./2)

            self.log("grad_norm", grad_norm)
