from typing import Iterable

import torch
from einops import rearrange
from torch import nn, Tensor
from torchaudio.transforms import Spectrogram
from toolz import juxt


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, window_sizes: Iterable[int]):
        # TODO: mel scale?
        self.stfts = nn.ModuleList(
            Spectrogram(
                n_fft=w,
                win_length=w,
                hop_length=w//4,
                normalized=False,
                power=1,
            )
            for w in window_sizes
        )

        self.eps = torch.tensor(1e-7)

    def _multiscale_stft(self, x: Tensor) -> Iterable[Tensor]:
        x_: Tensor = rearrange(x, "b c t -> (b c) t")
        # r.i.p. typing
        return juxt(self.stfts)(x_)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_spect = self._multiscale_stft(x)
        y_spect = self._multiscale_stft(y)

        loss = torch.tensor(0.)
        for xs, ys in zip(x_spect, y_spect):
            logx = torch.log(xs + self.eps)
            logy = torch.log(ys + self.eps)

            # TODO: different combinations of these distances?
            lin_distance = self._l2_relative_distance(xs, ys)
            log_distance = self._l1_distance(logx, logy)

            loss += lin_distance + log_distance

        return loss

    @staticmethod
    def _l1_distance(x: Tensor, y: Tensor) -> Tensor:
        return (x - y).abs().mean()

    @staticmethod
    def _l2_relative_distance(x: Tensor, y: Tensor) -> Tensor:
        return torch.pow(x - y, 2).mean() / torch.pow(y, 2).mean()
