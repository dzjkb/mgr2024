from typing import Iterable, cast

import torch
from einops import rearrange
from torch import nn, Tensor
from torchaudio.transforms import Spectrogram
from toolz import juxt


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, window_sizes: Iterable[int]):
        super().__init__()
        # TODO: mel scale?
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

        self.eps = torch.tensor(1e-7)

    def _multiscale_stft(self, x: Tensor) -> Iterable[Tensor]:
        x_: Tensor = rearrange(x, "b c t -> (b c) t")
        # r.i.p. typing
        return juxt(*self.stfts)(x_)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_spect = self._multiscale_stft(x)
        y_spect = self._multiscale_stft(y)

        loss = torch.tensor(0.0, device=x.device)
        for xs, ys in zip(x_spect, y_spect):
            logx = torch.log(xs + self.eps)
            logy = torch.log(ys + self.eps)

            # TODO: different combinations of these distances?
            lin_distance = _l2_relative_distance(xs, ys)
            log_distance = _l1_distance(logx, logy)

            loss += lin_distance + log_distance

        return loss


def _l2_relative_distance(x: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    return (torch.pow(x - y, 2).mean() + eps) / (torch.pow(y, 2).mean() + eps)


def _l1_distance(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).abs().mean()


def feature_matching_loss(real_fmaps: Tensor, fake_fmaps: Tensor) -> Tensor:
    return cast(
        Tensor,
        sum(
            [_l1_distance(real, fake) for real, fake in zip(real_fmaps, fake_fmaps)]
        ) / len(real_fmaps),
    )


def hinge_gan_losses(score_real:Tensor, score_fake: Tensor) -> tuple[Tensor, Tensor]:
    discriminator_loss = (torch.relu(cast(Tensor, 1 - score_real)) + torch.relu(cast(Tensor, 1 + score_fake))).mean()
    generator_loss = -score_fake.mean()
    return discriminator_loss, generator_loss
