from typing import cast

import torch
from einops import reduce


def make_mono(audio: torch.Tensor) -> torch.Tensor:
    assert len(audio.shape) == 3, f"got unexpected audio shape: {audio.shape}"
    mono_audio: torch.Tensor = cast(torch.Tensor, reduce(audio, "b c l -> b l", "mean"))
    return mono_audio
