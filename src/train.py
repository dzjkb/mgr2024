import yaml
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from attrs import frozen, asdict
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from .model import VAE, ModelConfig
from .dataset import AudioDataset
from .callbacks import init_callbacks, CallbacksConfig


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
    model: ModelConfig
    callbacks: CallbacksConfig


def _log_config(cfg: TrainingConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(yaml.dump(asdict(cfg)))


def _run_name(experiment_name: str) -> str:
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{experiment_name}_{time_str}"


def do_train(cfg: TrainingConfig) -> None:
    model = VAE(**asdict(cfg.model))
    train_set = DataLoader(
        AudioDataset(Path(cfg.train_dataset_path), transforms=[]),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_set = DataLoader(
        AudioDataset(Path(cfg.val_dataset_path), transforms=[]),
        batch_size=cfg.batch_size,
    )

    logger = TensorBoardLogger(save_dir=cfg.log_dir, name=_run_name(cfg.experiment_name))
    _log_config(cfg, f"{logger.log_dir}/config.yaml")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        devices=1,
        callbacks=init_callbacks(cfg.callbacks),
        profiler="simple",
        enable_progress_bar=True,
        # check_val_every_n_epoch=10,
    )
    checkpoint_kwarg: dict[str, str] = (
        {"ckpt_path": cfg.checkpoint_path} if cfg.checkpoint_path is not None else {}
    )
    trainer.fit(model, train_set, val_set, **checkpoint_kwarg)  # type: ignore
