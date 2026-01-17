from pathlib import Path
from typing import Iterable, cast

import torch
import torchaudio as ta
from attrs import asdict
from cattrs import structure
from torch import Tensor

from .model import VAE
from .model import VAE
from .configs import parse_hydra_config
from .train import TrainingConfig

CONFIGS_ROOT = "/home/jp/mgr2024/configs"


def _random_z(n: int, latent_size: int, length: int, device: torch.device) -> Tensor:
    return torch.distributions.Normal(
        torch.zeros(size=(n, latent_size, length)),
        torch.ones(size=(n, latent_size, length)),
    ).sample().to(device)


def _iterate_in_batches(x: Tensor, batch_size: int) -> Iterable[Tensor]:
    for idx in range(0, x.shape[0], batch_size):
        yield x[idx:idx + batch_size]


def do_generate(
    config_path: str,
    checkpoint: str,
    count: int,
    target_dir: str,
    batch_size: int | None,
    device: str | None,
) -> None:
    """
    generates `count` flac files in `target_dir` named `{idx}.flac` for `idx` in `range(count)` by drawing
    a random `z` from the prior and running it through the decoder
    """

    batch_size = batch_size or 8
    device = device or "cuda"

    cfg = structure(
        parse_hydra_config(CONFIGS_ROOT, config_path),
        TrainingConfig,
    )
    assert cfg.model.fixed_length is not None, "unconditional generation is available only for models with `fixed_length` defined"
    fixed_len = cast(int, cfg.model.fixed_length)

    device = torch.device(device)
    model = VAE.load_from_checkpoint(checkpoint, noise_config=cfg.noise, **asdict(cfg.model)).to(device)
    model.eval()

    generated_audio_list = []
    with torch.no_grad():
        z = _random_z(count, cfg.model.latent_size, fixed_len, device)
        for z_batch in _iterate_in_batches(z, batch_size=batch_size):
            generated = model._join_bands(model.decoder(z_batch))
            generated_audio_list.append(generated)
    
    generated_audio = torch.concat(generated_audio_list, dim=0)
    path = Path(target_dir)
    path.mkdir(parents=True)
    for i in range(generated_audio.shape[0]):
        ta.save(path / f"{i}.flac", generated_audio[i], cfg.dataset.expected_sample_rate)
