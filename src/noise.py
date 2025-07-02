from itertools import chain

import torch
from attrs import frozen
from torch import nn, Tensor, fft
from einops import rearrange


@frozen
class NoiseConfig:
    strides: list[int]
    latent_size: int
    n_filters: int


class Noise(nn.Module):
    def __init__(
        self,
        in_channels: int,
        strides: list[int],
        latent_size: int,
        n_filters: int,
    ):
        super().__init__()
        assert all(s  % 2 == 0 for s in strides), f"all strides should be even, got {strides}"
        assert len(strides) >= 2, f"at least length 2 strides required, got {strides}"

        self.n_filters = n_filters
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels,
                latent_size,
                kernel_size=2 * strides[0],
                padding=strides[0] // 2,
                stride=strides[0],
            ),
            *chain.from_iterable(
                (
                    nn.LeakyReLU(),
                    nn.Conv1d(
                        latent_size,
                        latent_size,
                        kernel_size=2 * stride,
                        padding=stride // 2,
                        stride=stride,
                    )
                )
                for stride in strides[1:-1]
            ),
            nn.Conv1d(
                latent_size,
                n_filters * in_channels,
                kernel_size=2 * strides[-1],
                padding=strides[-1] // 2,
                stride=strides[-1],
            ),
        )

        self.register_buffer(
            "frame_size", torch.prod(torch.tensor(strides))
        )
    
    @staticmethod
    def _scale_sigmoid(x: Tensor) -> Tensor:
        return 2 * torch.sigmoid(x - 5) ** 2.3 + 1e-7
    
    @staticmethod
    def _impulse_response(magnitudes: Tensor, frame_size: int) -> Tensor:
        zero_phase = torch.view_as_complex(torch.stack([magnitudes, torch.zeros_like(magnitudes)], -1))
        irs = torch.fft.irfft(zero_phase)
        ir_length = irs.shape[-1]

        shifted = torch.roll(irs, ir_length // 2, -1)  # puts the IR in the middle
        windowed = shifted * torch.hann_window(ir_length, dtype=shifted.dtype, device=shifted.device)
        padded_to_frame = torch.nn.functional.pad(windowed, (0, frame_size - int(ir_length)))
        return torch.roll(padded_to_frame, -ir_length // 2, -1)
    
    @staticmethod
    def _convolve(audio: Tensor, irs: Tensor) -> Tensor:
        assert list(audio.shape) == list(irs.shape)

        audio_padded = nn.functional.pad(audio_padded, (0, audio_padded.shape[-1]))
        irs_padded = nn.functional.pad(irs_padded, (irs_padded.shape[-1], 0))

        convolved = fft.irfft(fft.rfft(audio_padded) * fft.rfft(irs_padded))
        return convolved[..., convolved.shape[-1] // 2:]

    def forward(self, x: Tensor) -> Tensor:
        magnitudes = self._scale_sigmoid(self.net(x))
        irs = self._impulse_response(
            rearrange(magnitudes, "b (c filters) n_frames -> b n_frames c filters", filters=self.n_filters),
            int(self.frame_size),
        )

        # this automatically creates a noise signal split into frames of the same length as a single impulse response
        # the shape being (batch_size, n_frames, channels, frame_size)
        # each frame then gets convolved, preserving the shape, after which frames are concatenated to give the final
        # audio signal
        white_noise = torch.rand_like(irs) * 2 - 1  
        filtered_noise = self._convolve(white_noise, irs)
        return rearrange(filtered_noise, "b n_frames c frame_size -> b c (n_frames frame_size)")
