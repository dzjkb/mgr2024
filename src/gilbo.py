"""
GILBO — Generative Information Lower BOund (Alemi & Fischer, NeurIPS 2018, arXiv:1802.04874).

A dataset-free variational lower bound on the mutual information I(X; Z) under the model's own
generative joint p(z) p(x|z) (prior times decoder). We sample z from the prior, decode x = decoder(z)
with the decoder frozen, then train a fresh auxiliary Gaussian encoder e(z|x) to invert the decoder by
maximizing E[log e(z|x)]. The reported scalar is

    GILBO = E_{z ~ p(z), x = decoder(z)} [ log e(z|x) - log p(z) ]   (in nats)

The bound is valid for any e(z|x); training only tightens it.
"""

import math

import torch
from attrs import frozen
from torch import Tensor, optim

from .model import VAE, Encoder, ModelConfig
from .prior_model import Prior

_LOG_EVERY = 500


@frozen
class GilboConfig:
    train_steps: int = 20_000
    eval_steps: int = 2_000
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda"


def _build_aux_encoder(model_config: ModelConfig) -> Encoder:
    """A fresh, untrained copy of the VAE encoder architecture; the only trainable network in GILBO."""
    return Encoder(
        start_channels=model_config.capacity,
        dilations=model_config.dilations,
        strides=model_config.strides,
        latent_size=model_config.latent_size,
        do_weight_norm=model_config.do_weight_norm,
        activation=model_config.activation,
        n_bands=model_config.n_bands,
        audio_channels=1 if model_config.mono else 2,
        fixed_length=model_config.fixed_length,
        reduction_method=model_config.reduction_method,
    )


def _sample_prior(prior: Prior | None, n: int, latent_size: int, device: torch.device) -> Tensor:
    """Draw z ~ p(z) shaped (n, latent_size, 1), matching the decoder's input (see generate_with_model)."""
    if prior is None:
        return torch.randn(n, latent_size, 1, device=device)
    return prior.flow().sample((n,)).unsqueeze(-1).to(device)


def _log_pz(z: Tensor, prior: Prior | None) -> Tensor:
    """Per-sample log p(z), summed over the latent dimension. Shape (batch,)."""
    if prior is None:
        return torch.distributions.Normal(0.0, 1.0).log_prob(z).sum(dim=(1, 2))
    return prior.flow().log_prob(z.squeeze(-1))


def _gilbo_batch(aux_encoder: Encoder, model: VAE, prior: Prior | None, z: Tensor) -> Tensor:
    """One batch of the per-sample bound log e(z|x) - log p(z). Gradients flow only into aux_encoder."""
    with torch.no_grad():
        x = model._join_bands(model.decoder(z))
    x_mb = model._split_bands(x)
    mean, scale = aux_encoder(x_mb)
    std = VAE._std(scale)
    log_e = torch.distributions.Normal(mean, std).log_prob(z).sum(dim=(1, 2))
    return log_e - _log_pz(z, prior)


def compute_gilbo(
    model: VAE,
    prior: Prior | None,
    model_config: ModelConfig,
    cfg: GilboConfig,
) -> float:
    """Train the auxiliary encoder, then report the mean GILBO (in nats) over fresh samples."""
    device = torch.device(cfg.device)
    latent_size = model.latent_size

    model.eval()
    model.decoder.requires_grad_(False)
    if prior is not None:
        prior.eval()
        prior.requires_grad_(False)

    aux_encoder = _build_aux_encoder(model_config).to(device).train()
    optimizer = optim.Adam(aux_encoder.parameters(), lr=cfg.lr)

    running = 0.0
    for step in range(1, cfg.train_steps + 1):
        z = _sample_prior(prior, cfg.batch_size, latent_size, device)
        gilbo = _gilbo_batch(aux_encoder, model, prior, z).mean()
        loss = -gilbo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += gilbo.item()
        if step % _LOG_EVERY == 0:
            print(
                f"[gilbo] step {step}/{cfg.train_steps}  running GILBO = {running / _LOG_EVERY:.3f} nats"
            )
            running = 0.0

    aux_encoder.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(cfg.eval_steps):
            z = _sample_prior(prior, cfg.batch_size, latent_size, device)
            batch = _gilbo_batch(aux_encoder, model, prior, z)
            total += torch.nan_to_num(batch).sum().item()
            count += batch.shape[0]

    gilbo_nats = total / count
    print(f"[gilbo] GILBO = {gilbo_nats:.4f} nats ({gilbo_nats / math.log(2):.4f} bits)")
    return gilbo_nats
