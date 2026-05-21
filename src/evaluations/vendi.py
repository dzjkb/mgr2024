from torch import Tensor
from vendi_score import vendi

from ..ds_utils import load_audio_tensor


def vendi_score(embeddings: Tensor) -> float:
    """
    calculates the vendi score (see https://arxiv.org/pdf/2210.02410) for a set
    of embeddings shaped (n, d)
    """

    return vendi.score_dual(embeddings, normalize=True)


def vendi_for_serialized_embeddings(target_path: str) -> float:
    embeddings = load_audio_tensor(target_path).data
    return vendi_score(embeddings)
