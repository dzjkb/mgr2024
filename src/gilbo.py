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
from collections import deque

import torch
from attrs import frozen
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor, nn, optim

from .model import VAE, Encoder, ModelConfig
from .prior_model import Prior

_TREND_WINDOW = 200  # steps averaged for the live/windowed running GILBO
_MAX_SNAPSHOTS = 20  # convergence-trace snapshots printed over the course of training


@frozen
class GilboConfig:
    train_steps: int = 20_000
    eval_steps: int = 2_000
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda"
    # cosine-anneal the lr from `lr` down to `lr * lr_min_ratio` over `train_steps`; settles the tail
    # to a stable value instead of crawling at a constant lr. Set cosine_decay=False to disable.
    cosine_decay: bool = True
    lr_min_ratio: float = 1e-2
    # early-stop when the convergence trace plateaus: stop once the windowed GILBO has improved by less
    # than `early_stop_nats` over the last `early_stop_patience` snapshots. Set patience<=0 to disable.
    early_stop_patience: int = 4
    early_stop_nats: float = 0.5
    # clip the aux-encoder gradient norm each step. The Gaussian log_e term blows up quadratically if
    # e(z|x)'s predicted std collapses on a dim, which causes a self-reinforcing runaway; clipping is
    # the standard cure. Set grad_clip<=0 to disable.
    grad_clip: float = 10.0


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
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train_steps, eta_min=cfg.lr * cfg.lr_min_ratio
        )
        if cfg.cosine_decay
        else None
    )

    console = Console()
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[gilbo]}"),
    )

    # diagnostics: track a windowed (not cumulative) running mean so the live value follows the trend,
    # snapshot it at intervals to persist a convergence trace, and count non-finite batches.
    window: deque[float] = deque(maxlen=_TREND_WINDOW)
    snapshots: list[tuple[int, float]] = []
    nan_train = 0
    n_snapshots = min(_MAX_SNAPSHOTS, cfg.train_steps)
    snapshot_every = max(1, cfg.train_steps // n_snapshots)

    with Progress(*progress_columns, console=console) as progress:
        train_task = progress.add_task("training e(z|x)", total=cfg.train_steps, gilbo="GILBO = …")
        for step in range(1, cfg.train_steps + 1):
            z = _sample_prior(prior, cfg.batch_size, latent_size, device)
            gilbo = _gilbo_batch(aux_encoder, model, prior, z).mean()
            loss = -gilbo
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(aux_encoder.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            value = gilbo.item()
            if not math.isfinite(value):
                nan_train += 1
            else:
                window.append(value)
            windowed = sum(window) / len(window) if window else float("nan")

            if step % snapshot_every == 0 or step == cfg.train_steps:
                snapshots.append((step, windowed))
                progress.console.print(
                    f"  [dim]step {step:>7}/{cfg.train_steps}: GILBO ≈ {windowed:8.3f} nats[/dim]"
                )
                if _has_plateaued(snapshots, cfg):
                    progress.update(train_task, completed=cfg.train_steps)
                    progress.console.print(
                        f"  [green]plateaued: windowed GILBO improved < {cfg.early_stop_nats} nats over the "
                        f"last {cfg.early_stop_patience} snapshots — stopping early at step {step}.[/green]"
                    )
                    break
            progress.update(train_task, advance=1, gilbo=f"GILBO = {windowed:.3f} nats")

        aux_encoder.eval()
        eval_task = progress.add_task("evaluating", total=cfg.eval_steps, gilbo="GILBO = …")
        total = 0.0
        count = 0
        nan_eval = 0
        with torch.no_grad():
            for _ in range(cfg.eval_steps):
                z = _sample_prior(prior, cfg.batch_size, latent_size, device)
                batch = _gilbo_batch(aux_encoder, model, prior, z)
                nan_eval += int((~torch.isfinite(batch)).sum().item())
                total += torch.nan_to_num(batch).sum().item()
                count += batch.shape[0]
                progress.update(eval_task, advance=1, gilbo=f"GILBO = {total / count:.3f} nats")

    gilbo_nats = total / count
    _report_diagnostics(console, snapshots, nan_train, nan_eval, latent_size)
    console.print(
        f"[bold green]GILBO = {gilbo_nats:.4f} nats "
        f"({gilbo_nats / math.log(2):.4f} bits)[/bold green]"
    )
    return gilbo_nats


def _has_plateaued(snapshots: list[tuple[int, float]], cfg: GilboConfig) -> bool:
    """True once the windowed GILBO has genuinely flattened over the last `patience` snapshots.

    A real plateau means the recent values barely move in EITHER direction (total spread <
    `early_stop_nats`). We must NOT stop merely because the value stopped *rising*: if it is
    oscillating or falling (large spread), e(z|x) is unstable, not converged — keep training and let
    the lr schedule rein it in. (This is the bug that previously stopped a diverging run at step 45k.)
    """
    if cfg.early_stop_patience <= 0 or len(snapshots) < cfg.early_stop_patience + 1:
        return False
    recent = [v for _, v in snapshots[-(cfg.early_stop_patience + 1) :] if math.isfinite(v)]
    if len(recent) < cfg.early_stop_patience + 1:
        return False
    return (max(recent) - min(recent)) < cfg.early_stop_nats


def _report_diagnostics(
    console: Console,
    snapshots: list[tuple[int, float]],
    nan_train: int,
    nan_eval: int,
    latent_size: int,
) -> None:
    """Print a convergence verdict so 'under-trained' (rising) is distinguishable from a diverging /
    looser bound (falling). GILBO is a lower bound on a non-negative MI; it should rise toward 0+ as
    e(z|x) trains. A falling trend means e(z|x) is getting *worse* than the prior — an optimization
    problem, not a property of the model."""
    finite = [v for _, v in snapshots if math.isfinite(v)]
    if len(finite) >= 2:
        first, last = finite[0], finite[-1]
        peak = max(finite)
        # A converged bound is on the order of the latent's information content (~ a few nats/dim at
        # most). Values far beyond that are definitionally a blown-up optimization, regardless of which
        # way the trend points — check magnitude before interpreting the trend's sign.
        sane_ceiling = 100.0 * latent_size
        delta = last - first
        tol = max(0.5, 0.02 * abs(first))  # scaled so tiny wiggles near convergence read as "flat"
        if abs(last) > sane_ceiling:
            verdict = (
                f"[red]GILBO magnitude is implausible (|{last:.3g}| nats ≫ ~{sane_ceiling:.0g} for "
                f"{latent_size} dims) — e(z|x) has DIVERGED. Lower --lr substantially and rerun; check "
                f"the NaN report below and that the prior/decoder pairing generates valid audio.[/red]"
            )
        elif last < peak - tol:
            # rose to a peak then regressed: the optimizer went unstable, not converged
            verdict = (
                f"[red]GILBO PEAKED then REGRESSED (peaked at {peak:.2f}, ended at {last:.2f} nats). "
                f"e(z|x) became UNSTABLE after climbing — lower --lr (try 1e-4) so the late phase "
                f"settles instead of overshooting; the peak is a better estimate than the final value.[/red]"
            )
        elif delta > tol:
            verdict = (
                f"[yellow]GILBO is still RISING ({first:.2f} → {last:.2f} nats over training). "
                f"Likely UNDER-TRAINED — increase --train_steps and/or --lr and rerun.[/yellow]"
            )
        elif delta < -tol:
            verdict = (
                f"[red]GILBO is FALLING ({first:.2f} → {last:.2f} nats over training). The bound is "
                f"getting LOOSER, not tighter — e(z|x) is diverging. Try a lower --lr; check for NaNs "
                f"below and that the prior/decoder pairing actually generates valid audio.[/red]"
            )
        else:
            verdict = f"[green]GILBO looks CONVERGED (flat at ≈{last:.2f} nats over the last snapshots).[/green]"
        console.print(verdict)

    if nan_train or nan_eval:
        console.print(
            f"[red]⚠ non-finite GILBO batches: {nan_train} during training, {nan_eval} samples during "
            f"eval (zeroed via nan_to_num). This biases the reported value — a sign e(z|x)'s scale "
            f"collapsed. Lower --lr or check inputs.[/red]"
        )
