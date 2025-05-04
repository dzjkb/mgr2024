from pathlib import Path

import torch
import torchaudio as ta
from torch.utils import data


class AudioDataset(data.Dataset):
    def __init__(self, dataset_dir: Path, transforms: list) -> None:
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.name = dataset_dir.name
        self.files = list(dataset_dir.glob("*.wav"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio, sr = ta
