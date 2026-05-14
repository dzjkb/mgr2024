import os
import sys
from typing import Iterable
from contextlib import contextmanager

import torch
from laion_clap import CLAP_Module

SAMPLING_RATE = 48000  # fixed for CLAP


@contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _resolve_device() -> torch.device:
    # Opt-in: set CLAP_DEVICE=cuda to run on GPU, or CLAP_DEVICE=auto to auto-detect.
    # Default is CPU to preserve prior behavior.
    requested = os.environ.get("CLAP_DEVICE", "cpu").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        assert torch.cuda.is_available(), f"CLAP_DEVICE={requested!r} requested but CUDA is not available"
        return torch.device(requested if ":" in requested else "cuda:0")
    raise ValueError(f"unrecognized CLAP_DEVICE={requested!r}; expected one of: cpu, cuda, cuda:N, auto")


class _CLAPSingleton:
    _model: CLAP_Module | None = None
    _device: torch.device | None = None

    @classmethod
    def get_device(cls) -> torch.device:
        if cls._device is None:
            cls._device = _resolve_device()
        return cls._device

    @classmethod
    def get_model(cls) -> CLAP_Module:
        if cls._model is None:
            device = cls.get_device()
            with _suppress_stdout():
                cls._model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
                cls._model.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")
        return cls._model


def _iterate_batched(x: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for idx in range(0, x.shape[0], batch_size):
        yield x[idx:idx + batch_size]


def get_embeddings(audio: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """
    note: works on mono only, `audio` should be of `(n, t)` shape
    """

    assert len(audio.shape) == 2, f"clap accepts only mono audio, got a tensor with {len(audio.shape)} dimensions: {tuple(audio.shape)}"
    model = _CLAPSingleton.get_model()
    device = _CLAPSingleton.get_device()

    n = audio.shape[0]
    out: torch.Tensor | None = None
    offset = 0
    with torch.inference_mode():
        for batch in _iterate_batched(audio, batch_size):
            batch_dev = batch.to(device, non_blocking=True)
            emb = model.get_audio_embedding_from_data(x=batch_dev, use_tensor=True)
            emb_cpu = emb.detach().to("cpu", non_blocking=False).to(torch.float32)
            if out is None:
                out = torch.empty((n, emb_cpu.shape[1]), dtype=torch.float32)
            out[offset:offset + emb_cpu.shape[0]] = emb_cpu
            offset += emb_cpu.shape[0]
            del batch, batch_dev, emb, emb_cpu

    assert out is not None, "audio was empty"
    return out
