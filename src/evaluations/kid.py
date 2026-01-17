import torch
import numpy as np

from .clap import get_embeddings, SAMPLING_RATE
from .mmd import mmd
from ..ds_utils import load_directory, load_audio_tensor


def _make_mono(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


def kid_audio(set_x: torch.Tensor, set_y: torch.Tensor, batch_size: int = 5000) -> float:
    """
    assumes `len(set_x) <= len(set_y)`

    calculates mmd's in batches of `batch_size` and then returns the average
    """

    with torch.no_grad():
        x_mono = _make_mono(set_x) if len(set_x.shape) == 3 else set_x
        y_mono = _make_mono(set_y) if len(set_y.shape) == 3 else set_y
        
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


def kid_embeddings(set_x: torch.Tensor, set_y: torch.Tensor, batch_size: int = 5000) -> float:
    """
    assumes `len(set_x) <= len(set_y)`

    calculates mmd's in batches of `batch_size` and then returns the average

    assumes `set_x` and `set_y` are already CLAP embeddings
    """

    mmd_distances = []

    for j in range(0, set_x.shape[0], batch_size):
        x_batch = set_x[j:j+batch_size, ...]
        y_batch = set_y[j:j+batch_size, ...]
        mmd_distances.append(mmd(y_batch, x_batch))

    return np.mean(mmd_distances)


def kid_for_audio_directories(set_x_path: str, set_y_path: str, target_length: float) -> float:
    set_x = load_directory(set_x_path, target_length, sr=SAMPLING_RATE, mono=True).data
    set_y = load_directory(set_y_path, target_length, sr=SAMPLING_RATE, mono=True).data
    return kid_audio(set_x, set_y)


def kid_for_serialized_tensors(set_x_path: str, set_y_path: str) -> float:
    set_x = load_audio_tensor(set_x_path).data
    set_y = load_audio_tensor(set_y_path).data
    return kid_audio(set_x, set_y)


def kid_for_serialized_embeddings(set_x_path: str, set_y_path: str) -> float:
    set_x = load_audio_tensor(set_x_path).data
    set_y = load_audio_tensor(set_y_path).data
    return kid_embeddings(set_x, set_y)
