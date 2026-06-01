import contextlib
from typing import Iterator

import normflows as nf
import torch
from torch import nn, Tensor


class _ClampedMLP(nf.nets.MLP):
    def forward(self, x):
        return super().forward(x).clamp(-3, 3)


def _init_flow_nets(latent_size: int, hidden_size: int) -> tuple[nf.nets.MLP, nf.nets.MLP]:
    layer_sizes = [latent_size, latent_size, latent_size]
    return (
        nf.nets.MLP(layer_sizes, leaky=0.01, init_zeros=True),
        # _ClampedMLP(layer_sizes, leaky=0.01, init_zeros=True)
        nf.nets.MLP(layer_sizes, leaky=0.01, output_fn="tanh", init_zeros=True),
    )


@contextlib.contextmanager
def _frozen(module: nn.Module) -> Iterator[None]:
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in module.parameters():
            p.requires_grad_(True)


class RealNVP(nn.Module):
    def __init__(self, latent_size: int, num_layers: int = 16, hidden_size: int = 16) -> None:
        super().__init__()
        b = Tensor([1 - (i % 2) for i in range(latent_size)])

        flows: list[nn.Module] = []
        batchnorms: list[nn.Module] = []
        for i in range(num_layers):
            mask = b if i % 2 == 0 else 1 - b
            flows.append(nf.flows.MaskedAffineFlow(mask, *_init_flow_nets(latent_size, hidden_size)))
            if i < num_layers - 1:
                batchnorms.append(nn.BatchNorm1d(latent_size))

        self.flows = nn.ModuleList(flows)
        self.batchnorms = nn.ModuleList(batchnorms)
        self.freeze_weights = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ctx = _frozen(self) if self.freeze_weights else contextlib.nullcontext()
        with ctx:
            ld = x.new_zeros(x.shape[0])
            for i, flow in enumerate(self.flows):
                x, ld_ = flow(x.squeeze(-1))
                ld += ld_
                if i < len(self.batchnorms):
                    x = self.batchnorms[i](x)

            x_unsqueezed = x[..., None]
            return x_unsqueezed, ld

    def kld(self, x: Tensor, base_distribution: torch.distributions.Distribution) -> Tensor:
        # see https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py - forward_kld
        log_q = torch.zeros(x.shape[0], device=x.device)
        z = x.squeeze(-1)
        for i in range(len(self.flows) - 1, -1, -1):
            z, ld = self.flows[i].inverse(z)
            log_q += ld
        log_q += base_distribution.log_prob(z).sum(-1)
        return -torch.mean(log_q)

    def freeze(self, value: bool) -> None:
        self.freeze_weights = value
