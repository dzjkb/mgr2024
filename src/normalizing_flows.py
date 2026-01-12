import normflows as nf
import torch
from torch import nn, Tensor


def _init_flow_nets(latent_size: int) -> tuple[nf.nets.MLP, nf.nets.MLP]:
    layer_sizes = [latent_size, 16, latent_size]
    return (
        nf.nets.MLP(layer_sizes, leaky=0.01, init_zeros=True),
        nf.nets.MLP(layer_sizes, leaky=0.01, output_fn="tanh", init_zeros=True),
    )


class RealNVP(nn.Module):
    def __init__(self, latent_size: int, num_layers: int = 16) -> None:
        super().__init__()
        b = Tensor([1 - (i % 2) for i in range(latent_size)])
        flows = [
            (
                nf.flows.MaskedAffineFlow(b, *_init_flow_nets(latent_size))
                if i % 2 == 0
                else nf.flows.MaskedAffineFlow(1 - b, *_init_flow_nets(latent_size))
            )
            for i in range(num_layers)
        ]

        self.flows = nn.ModuleList(flows)
        self.register_buffer("freeze_weights", torch.tensor(0))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ld = torch.zeros(x.shape[0], device=x.device)
        for flow in self.flows:
            x, ld_ = flow(x.squeeze())
            ld += ld_

        x_unsqueezed = x[..., None]
        if self.freeze_weights:  # type: ignore[has-type]
            x_unsqueezed = x_unsqueezed.detach()
            ld = ld.detach()

        return x_unsqueezed, ld

    def kld(self, x: Tensor, base_distribution: torch.distributions.Distribution) -> Tensor:
        # see https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py - forward_kld
        log_q = torch.zeros(x.shape[0], device=x.device)
        z = x.squeeze()
        for i in range(len(self.flows) - 1, -1, -1):
            z, ld = self.flows[i].inverse(z)
            log_q += ld
        log_q += base_distribution.log_prob(z).mean(-1)
        return -torch.mean(log_q)

    def freeze(self, value: bool) -> None:
        self.freeze_weights = torch.tensor(int(value), device=self.freeze_weights.device)  # type: ignore[has-type]
