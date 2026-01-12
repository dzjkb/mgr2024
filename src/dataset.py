import random
from pathlib import Path
from typing import TypeAlias
from typing import Callable as Fn

import torch
import torchaudio as ta
from attrs import frozen, evolve
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
from toolz import compose_left

from .augmentations import random_phase_mangle
from .evaluations.utils import make_mono

TransformType: TypeAlias = Fn[[torch.Tensor], torch.Tensor]


def _id_transform(x: torch.Tensor) -> torch.Tensor:
    return x


def _random_crop(length: int) -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        cut_point = random.randint(0, x.shape[-1] - length)
        return x[..., cut_point : cut_point + length]

    return _transform_f


def _phase_mangle(sr: int) -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        return random_phase_mangle(x, 20, 2000, 0.99, sr)

    return _transform_f


def _dequantize(bits: int) -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand_like(x) / 2**bits 

    return _transform_f


def _zero_pad_cut(length: int) -> TransformType:
    assert length > 0, f"zero_pad_cut transform got invalid length {length}"
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        to_pad = length - x.shape[-1]
        if to_pad <= 0:
            return x[..., :length]
        return F.pad(x, (0, to_pad))

    return _transform_f


def _make_stereo() -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return torch.concat((x, x))
        return x

    return _transform_f


def _make_mono() -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0, keepdim=True)

    return _transform_f


def _cast_float() -> TransformType:
    def _transform_f(x: torch.Tensor) -> torch.Tensor:
        return x.type(dtype=torch.float32)

    return _transform_f


@frozen
class DatasetConfig:
    expected_sample_rate: int
    zero_pad_cut: int | None
    augment: bool
    mono: bool

    def val_overrides(self) -> "DatasetConfig":
        return evolve(self, augment=False)


class AudioDataset(data.Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_dir: Path,
        # transforms: list[TransformType],
        expected_sample_rate: int = 48000,
        zero_pad_cut: int | None = None,
        augment: bool = False,
        mono: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.name = dataset_dir.name
        self.expected_sample_rate = expected_sample_rate
        self.files = list(dataset_dir.glob("*.wav"))
        self.files_cache: dict[int, torch.Tensor] = dict()

        for idx in tqdm(range(len(self.files)), desc=f"validating the {dataset_dir.name} dataset"):
            assert self._validate(*self._load_file(idx)), f"invalid file: {self.files[idx]}"

        augmentations = [] if not augment else [
            _phase_mangle(expected_sample_rate),
            _dequantize(16),
        ]

        self.transforms = compose_left(
            *[
                _zero_pad_cut(zero_pad_cut) if zero_pad_cut is not None else _id_transform,
                _make_stereo(),  # some of the wavs are mono
                *augmentations,
                _make_mono() if mono else _id_transform,
                _cast_float(),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        returns a (channels, samples) length tensor
        """

        if index in self.files_cache:
            return self.files_cache[index]

        audio, _ = self._load_file(index)
        transformed = self.transforms(audio)
        self.files_cache[index] = transformed
        return transformed

    def _load_file(self, index: int) -> tuple[torch.Tensor, int]:
        """
        returns a (audio tensor, sample_rate) tuple
        """

        return ta.load(self.files[index])

    def _validate(self, audio: torch.Tensor, sample_rate: int) -> bool:
        return sample_rate == self.expected_sample_rate
