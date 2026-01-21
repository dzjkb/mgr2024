import json
from pathlib import Path

import torch
import torchaudio as ta
from attrs import frozen, evolve
from torch import Tensor
from torch.nn import functional as F

from .evaluations.clap import get_embeddings, SAMPLING_RATE


def save_generated_audio(audio: Tensor, target_dir: str, sample_rate: int) -> None:
    path = Path(target_dir)
    path.mkdir(parents=True)
    for i in range(audio.shape[0]):
        ta.save(path / f"{i}.flac", audio[i].cpu(), sample_rate)


def make_stereo(x: Tensor) -> Tensor:
    if x.shape[0] == 1:
        return torch.concat((x, x))
    return x


def make_mono(x: Tensor) -> Tensor:
    return x.mean(dim=0, keepdim=True)


def zero_pad_cut(x: Tensor, length: int) -> Tensor:
    to_pad = length - x.shape[-1]
    if to_pad <= 0:
        return x[..., :length]
    return F.pad(x, (0, to_pad))


def load_file(path: str, target_length_seconds: float, sr: int = 48000, mono: bool = True) -> Tensor:
    """
    returns a tensor of shape `(1, target_length_seconds * sr) if mono else (2, target_length_seconds * sr)`
    """

    audio, file_sr = ta.load(path)
    assert file_sr == sr
    target_length = int(target_length_seconds * sr)
    if mono:
        return zero_pad_cut(make_mono(audio), target_length).type(dtype=torch.float32)
    else:
        return zero_pad_cut(make_stereo(audio), target_length).type(dtype=torch.float32)


@frozen
class AudioTensor:
    data: Tensor
    source_dir: str
    filenames: list[str]


def load_directory(path: str, target_length_seconds: float, sr: int = 48000, mono: bool = True) -> AudioTensor:
    """
    note: doesn't search for audio files recursively
    returns a (file_count, channels, length) tensor wrapped in the `AudioTensor` class
    """

    files = [f for f in Path(path).iterdir() if f.is_file()]
    audio_tensor = torch.stack(
        [
            load_file(str(f), target_length_seconds=target_length_seconds, sr=sr, mono=mono)
            for f in files
        ],
        dim=0,
    )
    return AudioTensor(
        data=audio_tensor,
        source_dir=path,
        filenames=[f.name for f in files],
    )


def save_audio_tensor(t: AudioTensor, target_path: str) -> None:
    """
    creates a `target_path` directory with two files:
    - data.pt
    - metadata.json
    that serialize the corresponding `AudioTensor` fields
    """

    path = Path(target_path)
    assert not path.exists(), f"trying to save audio tensor to an already exsting path: {target_path}"
    path.mkdir(parents=True)

    torch.save(t.data, path / "data.pt")
    with open(path / "metadata.json", "w") as f:
        json.dump({"source_dir": t.source_dir, "filenames": t.filenames}, f)


def load_audio_tensor(source_path: str) -> AudioTensor:
    path = Path(source_path)
    audio_tensor = ta.load(path / "data.pt")
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    
    return AudioTensor(
        data=audio_tensor,
        source_dir=metadata["source_dir"],
        filenames=metadata["filenames"],
    )


def embed_directory(path: str, target_length_seconds: float) -> AudioTensor:
    audio_tensor = load_directory(
        path,
        target_length_seconds=target_length_seconds,
        sr=SAMPLING_RATE,
        mono=True,
    )
    assert audio_tensor.data.shape[1] == 1, "clap embeddings accept only mono audio"
    embeddings = get_embeddings(audio_tensor.data.squeeze(1))
    return evolve(audio_tensor, data=embeddings)
