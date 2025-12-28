import normflows as nf
import torch
from torch import nn, Tensor


def _init_flow_nets(hidden_size: int) -> tuple[nf.nets.MLP, nf.nets.MLP]:
    layer_sizes = [hidden_size, 2 * hidden_size, hidden_size]
    return nf.nets.MLP(layer_sizes, init_zeros=True), nf.nets.MLP(layer_sizes, init_zeros=True)


class RealNVP(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 16) -> None:
        super().__init__()
        b = Tensor([1 - (i % 2) for i in range(hidden_size)])
        flows = [
            (
                nf.flows.MaskedAffineFlow(b, *_init_flow_nets(hidden_size))
                if i % 2 == 0
                else nf.flows.MaskedAffineFlow(1 - b, *_init_flow_nets(hidden_size))
            )
            for i in range(num_layers)
        ]

        self.flows = nn.ModuleList(flows)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ld = Tensor([0.0])
        for flow in self.flows:
            x, ld_ = flow(x)
            ld += ld_

        return x, ld

    def kld(self, x: Tensor, base_distribution: torch.distributions.Distribution) -> Tensor:
        # see https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py - forward_kld
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, ld = self.flows[i].inverse(z)
            log_q += ld
        log_q += base_distribution.log_prob(z)
        return -torch.mean(log_q)
