import yaml
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from attrs import frozen, asdict
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from .model import VAE, ModelConfig
from .dataset import AudioDataset, DatasetConfig
from .callbacks import init_callbacks, CallbacksConfig
from .noise import NoiseConfig


@frozen
class TrainingConfig:
    train_dataset_path: str
    val_dataset_path: str
    log_dir: str
    experiment_name: str
    epochs: int
    batch_size: int
    device: str
    checkpoint_path: str | None
    separate_run: bool | None
    model: ModelConfig
    callbacks: CallbacksConfig
    noise: NoiseConfig
    dataset: DatasetConfig


def _log_config(cfg: TrainingConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml.dump(asdict(cfg)))


def _get_version(path: str) -> int:
    for part in path.split("/"):
        if part.startswith("version_"):
            return int(part.split("_")[1])
    assert False, f"checkpoint {path} is specified but no `version_x` folder found"


def do_train(cfg: TrainingConfig) -> None:
    if cfg.model.fixed_length is not None:
        assert cfg.model.fixed_length == cfg.dataset.zero_pad_cut, f"model must be configured to work on the same length as the dataset is padded to, got model length: {cfg.model.fixed_length}, dataset length: {cfg.dataset.zero_pad_cut}"
    model = VAE(noise_config=cfg.noise, **asdict(cfg.model))
    train_set = DataLoader(
        AudioDataset(Path(cfg.train_dataset_path), **asdict(cfg.dataset)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_set = DataLoader(
        AudioDataset(Path(cfg.val_dataset_path), **asdict(cfg.dataset)),
        batch_size=cfg.batch_size,
        num_workers=4,
    )

    if cfg.checkpoint_path is not None and not cfg.separate_run:
        logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name, version=_get_version(cfg.checkpoint_path))
    else:
        logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name)
    _log_config(cfg, f"{logger.log_dir}/config.yaml")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        devices=1,
        callbacks=init_callbacks(cfg.callbacks),
        profiler="simple",
        enable_progress_bar=True,
        check_val_every_n_epoch=100,
    )
    checkpoint_kwarg: dict[str, str] = (
        {"ckpt_path": cfg.checkpoint_path} if cfg.checkpoint_path is not None else {}
    )
    trainer.fit(model, train_set, val_set, **checkpoint_kwarg)  # type: ignore
