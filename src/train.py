import yaml
from datetime import datetime

import pytorch_lightning as pl
from attrs import frozen, asdict
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from .model import VAE, ModelConfig
from .dataset import AudioDataset


@frozen
class TrainingConfig:
    train_dataset_path: str
    val_dataset_path: str
    log_dir: str
    experiment_name: str
    batch_size: int
    model_config: ModelConfig


def _log_config(cfg: TrainingConfig, path: str) -> None:
    with open(path, "w") as f:
        f.write(yaml.dump(asdict(cfg)))


def _run_name(experiment_name: str) -> str:
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{experiment_name}_{time_str}"


def do_train(cfg: TrainingConfig) -> None:
    _log_config(cfg, f"{cfg.log_dir}/config.yaml")

    model = VAE(**asdict(cfg.model_config))
    train_set = DataLoader(
        AudioDataset(cfg.train_dataset_path),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_set = DataLoader(
        AudioDataset(cfg.val_dataset_path),
        batch_size=cfg.batch_size,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=cfg.log_dir, name=_run_name(cfg.experiment_name)),
        max_epochs=10,
    )
    trainer.fit(model, train_set, val_set)
    # trainer.fit(model, train_set, val_set, ckpt_path=run)
