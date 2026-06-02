from typing import Iterable, cast

import torch
from einops import rearrange
from torch import nn, Tensor
from torchaudio.transforms import Spectrogram, MelSpectrogram
from toolz import juxt


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        window_sizes: Iterable[int],
        eps: float = 1e-7,
        mel: bool = False,
        n_mels: int = 128,
        sample_rate: int = 48000,
    ):
        super().__init__()
        if mel:
            # Empty (all-zero) mel filters appear when a triangular filter is
            # narrower than the FFT bin spacing (sample_rate / n_fft). This bites
            # twice: at low frequencies (filters are densest there) and for small
            # windows (coarse bin spacing). f_min lifts the low edge above the
            # first-bin resolution, and n_mels is capped per-window so small
            # windows get proportionally fewer bands. Both together avoid the
            # all-zero-filterbank warning across every window size.
            self.stfts = nn.ModuleList(
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=w,
                    win_length=w,
                    hop_length=w // 4,
                    f_min=sample_rate / w,
                    n_mels=min(n_mels, w // 8),
                    normalized=False,
                    power=1,
                )
                for w in window_sizes
            )
        else:
            self.stfts = nn.ModuleList(
                Spectrogram(
                    n_fft=w,
                    win_length=w,
                    hop_length=w // 4,
                    normalized=False,
                    power=1,
                )
                for w in window_sizes
            )
        self.eps = eps

    # def _multiscale_stft(self, x: Tensor) -> Iterable[Tensor]:
    #     x_: Tensor = rearrange(x, "b c t -> (b c) t")
    #     # r.i.p. typing
    #     return juxt(*self.stfts)(x_)

    # def forward(self, x: Tensor, y: Tensor) -> Tensor:
    #     x_spect = self._multiscale_stft(x)
    #     y_spect = self._multiscale_stft(y)

    #     loss = x.new_zeros(1)
    #     for xs, ys in zip(x_spect, y_spect):
    #         logx = torch.log(xs + 1e-7)
    #         logy = torch.log(ys + 1e-7)

    #         # TODO: different combinations of these distances?
    #         lin_distance = _l2_relative_distance(xs, ys)
    #         log_distance = _l1_distance(logx, logy)

    #         loss += lin_distance + log_distance

    #     return loss

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = x.new_zeros(1)
        x_ = rearrange(x, "b c t -> (b c) t")
        y_ = rearrange(y, "b c t -> (b c) t")

        for stft in self.stfts:
            xs = stft(x_)
            ys = stft(y_)
            logx = torch.log(xs + self.eps)
            logy = torch.log(ys + self.eps)

            lin_distance = _l2_relative_distance(xs, ys)
            log_distance = _l1_distance(logx, logy)

            loss += lin_distance + log_distance

        return loss


def _l2_relative_distance(x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    return (torch.pow(x.sub(y), 2).mean() + eps) / (torch.pow(y, 2).mean() + eps)


def _l1_distance(x: Tensor, y: Tensor) -> Tensor:
    return x.sub(y).abs().mean()


def feature_matching_loss(real_fmaps: Tensor, fake_fmaps: Tensor) -> Tensor:
    return torch.stack([_l1_distance(real, fake) for real, fake in zip(real_fmaps, fake_fmaps)]).mean()


def hinge_gan_losses(score_real:Tensor, score_fake: Tensor) -> tuple[Tensor, Tensor]:
    discriminator_loss = (
        torch.relu(cast(Tensor, 1 - score_real))
        + torch.relu(cast(Tensor, 1 + score_fake))
    ).mean()
    generator_loss = -score_fake.mean()
    return discriminator_loss, generator_loss


if torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] >= 7):
    feature_matching_loss = torch.compile(feature_matching_loss)
    hinge_gan_losses = torch.compile(hinge_gan_losses)
