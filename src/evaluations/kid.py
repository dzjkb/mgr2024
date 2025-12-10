from typing import cast

import torch
import numpy as np
from einops import rearrange, reduce

from .clap import get_embeddings
from .mmd import mmd


def _make_mono(audio: torch.Tensor) -> torch.Tensor:
    assert len(audio.shape) == 3, f"got unexpected audio shape: {audio.shape}"
    mono_audio: torch.Tensor = cast(torch.Tensor, reduce(audio, "b c l -> b l", "mean")).cpu()
    return mono_audio


def kid(set_x: torch.Tensor, set_y: torch.Tensor, batch_size: int = 5000) -> float:
    """
    assumes `len(set_x) <= len(set_y)`

    calculates mmd's in batches of `batch_size` and then returns the average
    """

    with torch.no_grad():
        x_mono = _make_mono(set_x)
        y_mono = _make_mono(set_y)
        
        mmd_distances = []

        for j in range(0, set_x.shape[0], batch_size):
            x_batch = x_mono[j:j+batch_size, ...]
            y_batch = y_mono[j:j+batch_size, ...]

            x_embeddings_list = []
            y_embeddings_list = []
            embedding_batch_size = 32
            for i in range(0, x_batch.shape[0], embedding_batch_size):
                x_embeddings_list.append(get_embeddings(x_batch[i:i+embedding_batch_size]))
                y_embeddings_list.append(get_embeddings(y_batch[i:i+embedding_batch_size]))
            x_embeddings = torch.cat(x_embeddings_list, dim=0)
            y_embeddings = torch.cat(y_embeddings_list, dim=0)
            mmd_distances.append(mmd(y_embeddings, x_embeddings))

        return np.mean(mmd_distances)
