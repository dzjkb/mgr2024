from typing import Iterable

import torch
from torch import Tensor

from .model import VAE


def _random_z(n: int) -> Tensor:
    # TODO: sample a normal distribution
    return torch.tensor(0)


def _iterate_batched(x: Tensor, batch_size: int) -> Iterable[Tensor]:
    for idx in range(0, x.shape[0], batch_size):
        yield x[idx:idx + batch_size]


def do_generate(checkpoint: str, count: int, target_dir: str, batch_size: int = 4) -> None:
    model = VAE.load_from_checkpoint(checkpoint)
    model.eval()

    generated_audio_list = []
    with torch.no_grad():
        z = _random_z(count)
        for z_batch in _iterate_batched(z, batch_size=batch_size):
            generated = model.decoder(z_batch)  # TODO: gpu n shit?
            generated_audio_list.append(generated)
    
    generated_audio = torch.concat(generated_audio_list, dim=0)

    # TODO: save the results
