from pathlib import Path
from typing import TypeAlias, Any

import torch
import torchaudio as ta
from torch.utils import data
from tqdm import tqdm

# TODO
TransformType: TypeAlias = Any


def _id_transform(x: torch.Tensor) -> torch.Tensor:
    return x


class AudioDataset(data.Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_dir: Path,
        transforms: list[TransformType],
        expected_sample_rate: int = 48000,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.name = dataset_dir.name
        self.expected_sample_rate = expected_sample_rate
        self.files = list(dataset_dir.glob("*.wav"))

        for idx in tqdm(range(len(self.files)), desc=f"validating the {dataset_dir.name} dataset"):
            assert self._validate(*self._load_file(idx)), f"invalid file: {self.files[idx]}"

        # TODO compose transforms
        # self.transforms = compose(transforms)
        self.transforms = _id_transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        returns a (channels, samples) length tensor
        """

        audio, _ = self._load_file(index)
        return self.transforms(audio)
    
    def _load_file(self, index: int) -> tuple[torch.Tensor, int]:
        """
        returns a (audio tensor, sample_rate) tuple
        """

        return ta.load(self.files[index])
    
    def _validate(self, audio: torch.Tensor, sample_rate: int) -> bool:
        return sample_rate == self.expected_sample_rate
