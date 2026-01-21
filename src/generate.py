from typing import cast

import torch
from attrs import asdict

from .model import VAE, ModelConfig, generate_with_model
from .noise import NoiseConfig
from .ds_utils import save_generated_audio


def do_generate(
    model_config: ModelConfig,
    noise_config: NoiseConfig | None,
    sample_rate: int,
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

    assert model_config.fixed_length is not None, "unconditional generation is available only for models with `fixed_length` defined"

    device = torch.device(device)
    model = VAE.load_from_checkpoint(checkpoint, noise_config=noise_config, **asdict(model_config)).to(device)
    model.eval()

    audio = generate_with_model(model, count, model_config.latent_size, batch_size)
    save_generated_audio(audio, target_dir, sample_rate)

