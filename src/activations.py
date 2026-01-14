from typing import Callable

import torch
from torch import Tensor, jit, nn

ACTIVATIONS: dict[str, Callable[[int], nn.Module]] = {
    "relu": lambda dims: nn.LeakyReLU(.2),
    "snake": lambda dims: Snake1d(dims),
}


@jit.script  # type: ignore[attr-defined]
def snake(x: Tensor, alpha: Tensor) -> Tensor:
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)  # type: ignore


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels, 1))  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.alpha)
