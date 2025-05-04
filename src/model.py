from math import floor, ceil

from attrs import frozen, asdict

import pytorch_lightning as pl
import torch
from torch import optim, nn, Tensor


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

    half = stride / 2 + 1
    return floor(half), ceil(half)


@frozen
class ModelConfig:
    capacity: int
    latent_size: int
    dilations: list[list[int]]
    strides: list[int]
    latent_loss_weight: float = 0.5
    reconstruction_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        assert len(self.dilations) == len(self.strides), "strides and dilations need to be the same length (which is the desired amount of encoder/decoder blocks)"


class _ResidualDilatedUnit(nn.Module):
    kernel_size = 7

    def __init__(
        self,
        channels: int,
        dilation: int,
    ):
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
        self.net = nn.Sequential(
            *[
                _ResidualDilatedUnit(in_channels, in_channels, dilation=d)
                for d in dilations
            ],
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels,
                2*in_channels,
                kernel_size=2*stride,
                padding=_get_strided_padding(stride),
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
        assert len(dilations) == len(strides)
        self.latent_size = latent_size

        encoder_blocks = [
            _EncoderBlock(start_channels * 2**i, block_dilations, stride)
            for i, (block_dilations, stride) in enumerate(zip(dilations,strides))
        ]

        self.net = nn.Sequential(
            nn.Conv1d(
                2,  # left/right audio channel
                start_channels,
                kernel_size=self.input_kernel_size,
                padding=_get_padding(self.input_kernel_size),
            ),
            *encoder_blocks,
            nn.LeakyReLU(),
            nn.Conv1d(
                start_channels * 2**len(strides),
                latent_size * 2,  # outputs mean and log-variance for each dim
                kernel_size=self.output_kernel_size,
                padding=_get_padding(self.output_kernel_size),
            ),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.net(x)
        mean = encoded[:, :self.latent_size, :]
        logvar = encoded[:, self.latent_size:, :]
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
        # again, for simplicity - this will always be the case
        assert in_channels % 2 == 0

        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=2*stride,
                padding=_get_strided_padding(stride),
                stride=stride,
                output_padding=2,
                # TODO: alternatively don't set `output_padding` but set `padding` to just stride/2?
                # the output shape should stay the same
            ),
            *[
                _ResidualDilatedUnit(in_channels // 2, in_channels // 2, dilation=d)
                for d in dilations
            ],
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
    ):
        assert len(dilations) == len(strides)
        self.latent_size = latent_size

        decoder_blocks = [
            _DecoderBlock(start_channels / 2**i, block_dilations, stride)
            for i, (block_dilations, stride) in enumerate(zip(dilations,strides))
        ]

        self.net = nn.Sequential(
            nn.Conv1d(
                latent_size,
                start_channels,
                kernel_size=self.input_kernel_size,
                padding=_get_padding(self.input_kernel_size),
            ),
            *decoder_blocks,
            nn.LeakyReLU(),
            nn.Conv1d(
                start_channels / 2**len(strides),
                2,  # left/right audio channel
                kernel_size=self.output_kernel_size,
                padding=_get_padding(self.output_kernel_size),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class VAE(pl.LightningModule):
    def __init__(
        self,
        capacity: int,
        latent_size: int,
        dilations: list[list[int]],
        strides: list[int],
        latent_loss_weight: float = 0.5,
        reconstruction_loss_weight: float = 1.0,
    ):
        assert len(dilations) == len(strides)

        self.encoder = Encoder(
            start_channels=capacity,
            dilations=dilations,
            strides=strides,
            latent_size=latent_size,
        )
        self.decoder = Decoder(
            start_channels=capacity * 2**len(strides),
            dilations=dilations,
            strides=strides[::-1],
            latent_size=latent_size,
        )

        # std = nn.functional.softplus(scale) + 1e-4
        # var = std * std
        # logvar = torch.log(var)

        self.latent_loss_weight = latent_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def _reparametrize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        # kl divergence for latent space regularization
        # kl = (mean * mean + var - logvar - 1).sum(1).mean()
        # return z, self.beta * kl
        return z

    @staticmethod
    def _latent_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        # TODO: what exponent should `torch.exp()` have here, what dimensions?
        # TODO: if normalizing flows add `log_det`
        return (mean.pow(2) + torch.exp(logvar) - logvar - 1).sum(1).mean()

    @staticmethod
    def _reconstruction_loss(x: Tensor, x_hat: Tensor) -> Tensor:
        return x - x_hat  # TODO

    # def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
    #     # mean, logvar = self.subnet_mean(z), self.subnet_logvar(z)
    #     mean, logvar = self.encoder(x)
    #     z  = self._reparametrize(mean, logvar)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # encoder expects (N, 2, L)
        mean, logvar = self.encoder(x)
        z = self._reparametrize(mean, logvar)
        x_hat = self.decoder(z)

        loss = (
            self.latent_loss_weight * self._latent_loss(mean, logvar) +
            self.reconstruction_loss_weight * self._reconstruction_loss(x, x_hat)
        )
        return x_hat, loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        # batch is whatever the dataloader returns
        x_hat, loss = self.forward(batch)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer