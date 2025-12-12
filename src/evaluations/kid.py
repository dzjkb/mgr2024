from typing import cast

import torch
import numpy as np
from einops import reduce

from .clap import get_embeddings
from .mmd import mmd
from .utils import make_mono


def kid(set_x: torch.Tensor, set_y: torch.Tensor, batch_size: int = 5000) -> float:
    """
    assumes `len(set_x) <= len(set_y)`

    calculates mmd's in batches of `batch_size` and then returns the average
    """

    with torch.no_grad():
        x_mono = make_mono(set_x) if len(set_x.shape) == 3 else set_x
        y_mono = make_mono(set_y) if len(set_y.shape) == 3 else set_y
        
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
