from typing import Callable as Fn

import torch
from torch import Tensor
from attrs import asdict

from .model import VAE, ModelConfig, generate_with_model
from .noise import NoiseConfig
from .ds_utils import save_generated_audio, save_audio_tensor, embed_directory


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
    model = VAE.load_from_checkpoint(checkpoint, noise_config=noise_config, **asdict(model_config), map_location=device, weights_only=False)
    model.eval()

    audio = generate_with_model(model, count, model_config.latent_size, batch_size)
    save_generated_audio(audio, target_dir, sample_rate)


def _encode(model: VAE, x: Tensor) -> Tensor:
    with torch.inference_mode():
        x_mb = model._split_bands(x)
        # mean, logvar = self.encoder(x_mb)
        mean, scale = model.encoder(x_mb)
        z = model._reparametrize(mean, scale)

        if model.posterior is not None:
            z, _ = model.posterior(z)
        
    return z.squeeze()


def _reconstruct(model: VAE, x: Tensor) -> Tensor:
    with torch.inference_mode():
        x_mb = model._split_bands(x)
        # mean, logvar = self.encoder(x_mb)
        mean, scale = model.encoder(x_mb)
        z = model._reparametrize(mean, scale)

        if model.posterior is not None:
            z, _ = model.posterior(z)

        x_hat_mb = model.decoder(z)
        x_hat = model._join_bands(x_hat_mb)
        
    return x_hat.squeeze()


def _do_model_pass(
    model_config: ModelConfig,
    noise_config: NoiseConfig | None,
    sample_rate: int,
    checkpoint: str,
    source_dir: str,
    batch_size: int | None,
    device: str | None,
    forward_f: Fn[[VAE, Tensor], Tensor],
    output_suffix: str,
) -> None:
    """
    runs the files from `source_dir` through `forward_f` and saves the results in an `AudioTensor`
    """

    batch_size = batch_size or 8
    device = device or "cuda"

    assert model_config.fixed_length is not None, "forward passes are available only for models with `fixed_length` defined"

    device = torch.device(device)
    model = VAE.load_from_checkpoint(checkpoint, noise_config=noise_config, **asdict(model_config), map_location=device, weights_only=False)
    model.eval()

    def _encoding_f(x: Tensor, b_size: int) -> Tensor:
        return forward_f(model, x.unsqueeze(1).to(device))  # missing channel dimension, our VAE expects it even if it's 1

    result = embed_directory(source_dir, model_config.fixed_length / sample_rate, _encoding_f)
    save_audio_tensor(result, f"{source_dir}_{output_suffix}")


def do_encode(
    model_config: ModelConfig,
    noise_config: NoiseConfig | None,
    sample_rate: int,
    checkpoint: str,
    source_dir: str,
    batch_size: int | None,
    device: str | None,
    unique_suffix: str = "",
) -> None:
    _do_model_pass(
        model_config,
        noise_config,
        sample_rate,
        checkpoint,
        source_dir,
        batch_size,
        device,
        _encode,
        f"in_latent_space_{unique_suffix}",
    )


def do_reconstruct(
    model_config: ModelConfig,
    noise_config: NoiseConfig | None,
    sample_rate: int,
    checkpoint: str,
    source_dir: str,
    batch_size: int | None,
    device: str | None,
    unique_suffix: str = "",
) -> None:
    _do_model_pass(
        model_config,
        noise_config,
        sample_rate,
        checkpoint,
        source_dir,
        batch_size,
        device,
        _reconstruct,
        f"reconstructions_{unique_suffix}",
    )
