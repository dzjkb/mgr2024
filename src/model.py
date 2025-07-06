from math import floor, ceil

from attrs import frozen, asdict

import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
import torch
import torch.autograd
from einops import rearrange, reduce
from torch import optim, nn, Tensor

from .loss import MultiScaleSTFTLoss
from .plots import draw_histogram
from .noise import Noise, NoiseConfig

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
    adam_betas: tuple[float, float] = (0.5, 0.9)
    monitor_grad_norm: bool = False
    detect_nans: bool = False
    initial_lr: float = 1e-4
    final_lr: float = 1e-5

    def __post_init__(self) -> None:
        assert len(self.dilations) == len(
            self.strides
        ), "strides and dilations need to be the same length (which is the desired amount of encoder/decoder blocks)"


class _ResidualDilatedUnit(nn.Module):
    kernel_size = 7

    def __init__(
        self,
        channels: int,
        dilation: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=_get_padding(self.kernel_size, dilation),
            ),
            nn.LeakyReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
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
    ):
        super().__init__()
        # assert stride % 2 == 0, f"stride should be even, got {stride}"

        self.net = nn.Sequential(
            *[_ResidualDilatedUnit(in_channels, dilation=d) for d in dilations],
            nn.LeakyReLU(),
            # asymmetrical padding might be breaking something, try sym
            ZeroPad1d(*_get_strided_padding(stride)),
            nn.Conv1d(
                in_channels,
                2 * in_channels,
                kernel_size=2 * stride,
                padding=0,
                # padding=stride // 2,
                stride=stride,
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
    ):
        super().__init__()
        assert len(dilations) == len(strides)
        self.latent_size = latent_size

        encoder_blocks = [
            _EncoderBlock(start_channels * 2**i, block_dilations, stride)
            for i, (block_dilations, stride) in enumerate(zip(dilations, strides))
        ]

        self.net = nn.Sequential(
            nn.Conv1d(
                2,  # left/right audio channel
                start_channels,
                kernel_size=self.input_kernel_size,
                padding=_get_padding(self.input_kernel_size, 1),
            ),
            *encoder_blocks,
            nn.LeakyReLU(),
            nn.Conv1d(
                start_channels * 2 ** len(strides),
                latent_size * 2,  # outputs mean and log-variance for each dim
                kernel_size=self.output_kernel_size,
                padding=_get_padding(self.output_kernel_size, 1),
            ),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.net(x)
        mean = encoded[:, : self.latent_size, :]
        logvar = encoded[:, self.latent_size :, :]
        return mean, logvar


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
    ):
        super().__init__()
        # again, for simplicity - this will always be the case
        assert in_channels % 2 == 0

        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=2 * stride,
                padding=_get_strided_padding(stride)[1],
                stride=stride,
                output_padding=stride % 2,
            ),
            *[_ResidualDilatedUnit(in_channels // 2, dilation=d) for d in dilations],
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
    ):
        super().__init__()
        assert len(dilations) == len(strides)
        self.latent_size = latent_size

        decoder_blocks = [
            _DecoderBlock(start_channels // 2**i, block_dilations, stride)
            for i, (block_dilations, stride) in enumerate(zip(dilations, strides))
        ]

        self.net = nn.Sequential(
            nn.Conv1d(
                latent_size,
                start_channels,
                kernel_size=self.input_kernel_size,
                padding=_get_padding(self.input_kernel_size, 1),
            ),
            *decoder_blocks,
            nn.LeakyReLU(),
            # nn.Conv1d(
            #     start_channels // 2 ** len(strides),
            #     2,  # left/right audio channel
            #     kernel_size=self.output_kernel_size,
            #     padding=_get_padding(self.output_kernel_size, 1),
            # ),
        )

        self.waveform_amp_net = nn.Conv1d(
            start_channels // 2 ** len(strides),
            2 * 2,  # left/right audio channel + waveform/amplitude for each
            kernel_size=self.output_kernel_size,
            padding=_get_padding(self.output_kernel_size, 1),
        )

        self.noise_net = (
            Noise(
                in_channels=start_channels // 2 ** len(strides),
                **asdict(noise_config),
            )
            if noise_config is not None
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        x_hat = self.net(x)

        waveform_amp = self.waveform_amp_net(x_hat)
        waveform, amp_mod = torch.split(
            waveform_amp,
            waveform_amp.shape[1] // 2,
            dim=1,
        )
        amp_modded = waveform * torch.sigmoid(amp_mod)

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
        adam_betas: tuple[float, float] = (0.5, 0.9),
        lr_decay_steps: int = 10000,
        monitor_grad_norm: bool = False,
        detect_nans: bool = False,
        initial_lr: float = 1e-4,
        final_lr: float = 1e-5,
    ):
        super().__init__()
        assert len(dilations) == len(strides)

        self.encoder = Encoder(
            start_channels=capacity,
            dilations=dilations,
            strides=strides,
            latent_size=latent_size,
        )
        self.decoder = Decoder(
            start_channels=capacity * 2 ** len(strides),
            dilations=dilations,
            strides=strides[::-1],
            latent_size=latent_size,
            noise_config=noise_config,
        )

        _window_sizes = stft_window_sizes or [2048, 1024, 512, 256, 128]
        self.reconstruction_loss = MultiScaleSTFTLoss(_window_sizes)

        self.latent_loss_weight = latent_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.betas = adam_betas
        self.lr_decay_steps = lr_decay_steps
        self.monitor_grad_norm = monitor_grad_norm
        self.detect_nans = detect_nans
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.validation_outputs: dict[str, list[Tensor]] = {
            "audio": [],
            "latent": [],
            "original": [],
        }
        self.validation_epoch = 0

    def _reparametrize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)  # type: ignore
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    @staticmethod
    def _latent_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        # in the future - if normalizing flows add `log_det`
        return (mean.pow(2) + torch.exp(logvar) - logvar - 1).sum(1).mean()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        mean, logvar = self.encoder(x)
        z = self._reparametrize(mean, logvar)
        x_hat = self.decoder(z)

        losses_dict = {
            "latent_loss": self._latent_loss(mean, logvar),
            "reconstruction_loss": self.reconstruction_loss(x, x_hat),
        }
        total_loss = (
            self.latent_loss_weight * losses_dict["latent_loss"]
            + self.reconstruction_loss_weight * losses_dict["reconstruction_loss"]
        )
        return x_hat, z, total_loss, losses_dict

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        _, _, loss, losses_dict = self.forward(batch)
        self.log("loss/train_loss", loss, on_step=True, on_epoch=True)
        for loss_key, value_tensor in losses_dict.items():
            self.log(f"loss/{loss_key}", value_tensor, on_step=True, on_epoch=True)
        self.log("latent_loss_weight", self.latent_loss_weight)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        reconstructed_audio, z, loss, _ = self.forward(batch)
        self.log("loss/validation_loss", loss)

        self.validation_outputs["original"].append(batch)
        self.validation_outputs["audio"].append(reconstructed_audio)
        self.validation_outputs["latent"].append(z)

    @staticmethod
    def _mono_concatenate_batch(audio: Tensor) -> Tensor:
        assert len(audio.shape) == 3, f"got unexpected audio shape: {audio.shape}"
        mono_audio: Tensor = reduce(audio, "b c l -> b l", "mean")
        audio_concatenated: Tensor = rearrange(mono_audio, "b l -> (b l)").cpu()
        return audio_concatenated

    def on_validation_epoch_end(self) -> None:
        # TODO: maybe we want more?
        # validation_audio = rearrange(
        #     self.validation_outputs,
        #     "li b c len -> (li b) c len",
        # )
        # taking first batch for now
        validation_audio = self.validation_outputs["audio"][0]
        audio_concatenated = self._mono_concatenate_batch(validation_audio)
        self.logger.experiment.add_audio(  # type: ignore
            "validation_audio",
            audio_concatenated.numpy(),
            self.validation_epoch,
            SAMPLING_RATE,
        )
        if self.validation_epoch <= 0:
            original_audio = self.validation_outputs["original"][0]
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

        validation_embeddings = self.validation_outputs["latent"][0]
        embeddings_concatenated: Tensor = rearrange(validation_embeddings, "b d l -> (b l) d")
        assert len(validation_embeddings.shape) == 3, f"got unexpected embedding shape: {validation_embeddings.shape}"
        self.logger.experiment.add_embedding(
            embeddings_concatenated.cpu().numpy(),
            tag="latent space",
            global_step=self.validation_epoch,
        )  # type: ignore

        self.validation_outputs["audio"] = []
        self.validation_outputs["latent"] = []
        self.validation_outputs["original"] = []
        self.validation_epoch += 1

        if self.detect_nans:
            torch.autograd.set_detect_anomaly(True)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr, betas=self.betas)
        lr_schedule = optim.lr_scheduler.LinearLR(
            optimizer, 1.0, self.final_lr / self.initial_lr, self.lr_decay_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_schedule},
        }

    def on_train_batch_end(self, x_hat: Tensor, batch: Tensor, batch_idx: int) -> None:
        if self.monitor_grad_norm:
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in [*self.encoder.parameters(), *self.decoder.parameters()]
                if p.grad is not None
            ) ** (1./2)

            self.log("grad_norm", grad_norm)
