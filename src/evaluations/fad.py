# from fadtk import FrechetAudioDistance, CLAPLaionModel

# class _CLAPSingleton:
#     _model: CLAPLaionModel | None = None

#     @classmethod
#     def get_model(cls) -> CLAPLaionModel:
#         if cls._model is None:
#             cls._model = CLAPLaionModel("music")
#         return cls._model
#
# this needs torch>=2.3.0, maybe some day

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from fadtk import FrechetAudioDistance, ModelLoader, cache_embedding_files
from fadtk.fad import FADInfResults, calc_embd_statistics, calc_frechet_distance

from .clap import _CLAPSingleton, SAMPLING_RATE


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# FADTK_LOCATION = "/home/jp/fadtk"

class _CLAPModelLoader(ModelLoader):
    """
    Adapter that exposes the shared `_CLAPSingleton` (LAION CLAP, HTSAT-base,
    music checkpoint) to `fadtk.FrechetAudioDistance`. Embedding logic mirrors
    `fadtk.CLAPLaionModel._get_embedding`: 10s chunks with 1s hop, int16
    quantization, then `get_audio_embedding_from_data(use_tensor=True)`.
    """

    def __init__(self) -> None:
        super().__init__(name="clap-laion-music", num_features=512, sr=SAMPLING_RATE)
        self.device = _CLAPSingleton.get_device()

    def load_model(self) -> None:
        self.model = _CLAPSingleton.get_model()

    @staticmethod
    def _int16_to_float32(x: np.ndarray) -> np.ndarray:
        return (x / 32767.0).astype(np.float32)

    @staticmethod
    def _float32_to_int16(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        audio = audio.reshape(1, -1)
        audio = self._int16_to_float32(self._float32_to_int16(audio))

        chunk_size = 10 * self.sr  # 10s
        hop_size = self.sr  # 1s
        chunks = [audio[:, i:i + chunk_size] for i in range(0, audio.shape[1], hop_size)]

        embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for chunk in chunks:
                if chunk.shape[1] != chunk_size:
                    chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])))
                chunk_t = torch.from_numpy(chunk).float().to(self.device, non_blocking=True)
                emb = self.model.get_audio_embedding_from_data(x=chunk_t, use_tensor=True)
                embeddings.append(emb.detach())
        return torch.cat(embeddings, dim=0)  # (timeframes, 512)


# def fad(reference_set: str, target_set: str, fadtk_location: str = FADTK_LOCATION) -> float:
def fad(reference_set: str, target_set: str) -> float:
    """
    based on https://github.com/microsoft/fadtk/blob/main/fadtk/__main__.py
    calculates FAD_inf to correct for the sample size bias

    note: fadtk splits incoming audio into 10s frames and pads them, which might give a slight difference from
    using the `fad_from_embeddings()` function
    """

    # ref_set = str(Path(reference_set).resolve())
    # tgt_set = str(Path(target_set).resolve())
    # with cd(fadtk_location):
    #     fadtk_run = subprocess.run(
    #         ["CUDA_VISIBLE_DEVICES=\"\"", "uv", "run", "python", "-m", "fadtk", "clap-laion-audio", ref_set, tgt_set, "--inf"],
    #         text=True,
    #         capture_output=True,
    #     )
    #     fad_score = float(fadtk_run.stdout.split()[-1])
    # return fad_score

    model = _CLAPModelLoader()
    fad = FrechetAudioDistance(model)
    cache_embedding_files(reference_set)
    cache_embedding_files(target_set)
    score = fad.score_inf(reference_set, list(Path(target_set).glob('*.*')))
    return score.score


def _as_embedding_array(x) -> np.ndarray:
    """Accept torch.Tensor / np.ndarray / AudioTensor and return a 2D (n, d) float array."""
    if hasattr(x, "data") and not isinstance(x, (np.ndarray, torch.Tensor)):
        # duck-type AudioTensor (from src.ds_utils)
        x = x.data
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert x.ndim == 2, f"expected 2D (n_samples, n_features) embeddings, got shape {x.shape}"
    return x.astype(np.float64, copy=False)


def fad_from_embeddings(
    reference_embeddings,
    target_embeddings,
    steps: int = 25,
    min_n: int = 500,
) -> FADInfResults:
    """
    FAD-inf computed from precomputed embeddings. Mirrors `FrechetAudioDistance.score_inf`
    but skips all audio loading and per-file caching.

    Accepts numpy arrays, torch tensors, or `AudioTensor` instances. Embeddings must be
    shaped `(n_samples, n_features)` (e.g. the output of `get_embedding_directory` /
    `embed_directory`).
    """

    ref = _as_embedding_array(reference_embeddings)
    tgt = _as_embedding_array(target_embeddings)

    mu_base, cov_base = calc_embd_statistics(ref)

    max_n = tgt.shape[0]
    if min_n > max_n:
        min_n = max_n
    ns = [int(n) for n in np.linspace(min_n, max_n, steps)]

    results: list[list[float]] = []
    for n in ns:
        indices = np.random.choice(max_n, size=n, replace=True)
        embds_eval = tgt[indices]
        mu_eval, cov_eval = calc_embd_statistics(embds_eval)
        fad_score = calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)
        results.append([n, fad_score])

    ys = np.array(results)
    xs = 1 / np.array(ns)
    slope, intercept = np.polyfit(xs, ys[:, 1], 1)
    r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum(
        (ys[:, 1] - np.mean(ys[:, 1])) ** 2
    )

    return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)
