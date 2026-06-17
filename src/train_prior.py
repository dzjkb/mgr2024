from typing import Callable

import torch
import pytorch_lightning as pl
import yaml
from attrs import asdict, frozen
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

from .prior_model import Prior, PriorConfig
from .train import diagnosing_collate_fn
from .ds_utils import load_audio_tensor


@frozen
class PriorTrainingConfig:
    train_dataset_path: str
    log_dir: str
    experiment_name: str
    epochs: int
    batch_size: int
    device: str
    val_every: int
    model: PriorConfig
    checkpoint_path: str | None = None
    separate_run: bool | None = None
    precision: str | None = None


def _log_config(cfg: PriorTrainingConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml.dump(asdict(cfg)))


def _get_version(path: str) -> int:
    for part in path.split("/"):
        if part.startswith("version_"):
            return int(part.split("_")[1])
    assert False, f"checkpoint {path} is specified but no `version_x` folder found"


def do_train(cfg: PriorTrainingConfig) -> None:
    torch.manual_seed(420)
    model = Prior(**asdict(cfg.model))
    train_data = load_audio_tensor(cfg.train_dataset_path).data
    shuffled_train_data = train_data[torch.randperm(len(train_data))]
    split_idx = len(train_data) // 10
    train_set = DataLoader(
        TensorDataset(shuffled_train_data[:-split_idx]),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=diagnosing_collate_fn,
    )
    val_set = DataLoader(
        TensorDataset(shuffled_train_data[-split_idx:]),
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=diagnosing_collate_fn,
    )

    if cfg.checkpoint_path is not None and not cfg.separate_run:
        logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name, version=_get_version(cfg.checkpoint_path))
    else:
        logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name)
    _log_config(cfg, f"{logger.log_dir}/config.yaml")

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        devices=1,
        enable_progress_bar=True,
        check_val_every_n_epoch=cfg.val_every,
        precision=cfg.precision,
    )
    checkpoint_kwarg: dict[str, str] = (
        {"ckpt_path": cfg.checkpoint_path, "weights_only": False} if cfg.checkpoint_path is not None else {}
    )
    trainer.fit(model, train_set, val_set, **checkpoint_kwarg)  # type: ignore