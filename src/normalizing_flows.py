import contextlib
from typing import Iterator

from torch import nn, Tensor
from zuko.flows import RealNVP


@contextlib.contextmanager
def _frozen(module: nn.Module) -> Iterator[None]:
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in module.parameters():
            p.requires_grad_(True)


class RealNVPTransform(nn.Module):
    def __init__(self, latent_size: int, num_layers: int = 16, hidden_size: list[int] = [64, 64]) -> None:
        super().__init__()
        # we don't need the base distribution functionality
        self.flow_transform = RealNVP(latent_size, transforms=num_layers, hidden_features=hidden_size).transform
        self.freeze_weights = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ctx = _frozen(self) if self.freeze_weights else contextlib.nullcontext()
        with ctx:
            y, ladj = self.flow_transform().call_and_ladj(x.squeeze(-1))  # type: ignore
            return y[..., None], ladj

    def freeze(self, value: bool) -> None:
        self.freeze_weights = value
