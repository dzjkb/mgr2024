import os
import yaml
from typing import Any, Optional
from pathlib import Path

import click
import torchaudio as ta
from cattrs import structure
from tqdm import tqdm

from .train import do_train, TrainingConfig
from .model import ModelConfig, SAMPLING_RATE
from .callbacks import CallbacksConfig


def _push_to_subcommands(option: str, value: str, cfg: dict[str, Any]) -> None:
    for c in cfg:
        if not isinstance(cfg[c], dict):
            continue
        cfg[c][option] = value


@click.group(context_settings={"auto_envvar_prefix": "JPMGR"})
@click.option("--config", default="./default_config.yml", type=click.Path())
@click.pass_context
def jpmgr(ctx, config):  # type: ignore
    if os.path.exists(config):
        with open(config, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        if (dataset_dir := config.get("dataset_dir")) is not None:
            _push_to_subcommands("dataset_dir", dataset_dir, config)

        ctx.default_map = config


@jpmgr.command()
@click.option("--train_dataset_path", type=click.Path())
@click.option("--val_dataset_path", type=click.Path())
@click.option("--log_dir", type=click.Path())
@click.option("--experiment_name", type=str)
@click.option("--epochs", type=int)
@click.option("--batch_size", type=int)
@click.option("--device", type=str, default="cpu")
@click.option("--checkpoint_path", default=None, type=Optional[str])
@click.option("--model_config", type=click.Path())
@click.option("--callbacks_config", type=click.Path())
def train(
    train_dataset_path: str,
    val_dataset_path: str,
    log_dir: str,
    experiment_name: str,
    epochs: int,
    batch_size: int,
    device: str,
    checkpoint_path: str | None,
    model_config: str,
    callbacks_config: str,
) -> None:
    with (
        open(model_config) as mf,
        open(callbacks_config) as cf,
    ):
        subconfigs = [
            ("model_config", mf, ModelConfig),
            ("callbacks_config", cf, CallbacksConfig),
        ]

        subconfigs_dict: dict[str, ModelConfig | CallbacksConfig] = {
            name: structure(
                yaml.load(file_handle, Loader=yaml.SafeLoader),
                cfg_class,
            )
            for name, file_handle, cfg_class in subconfigs
        }

        cfg = TrainingConfig(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            log_dir=log_dir,
            experiment_name=experiment_name,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
            **subconfigs_dict,  # type: ignore
        )

    do_train(cfg)


@jpmgr.command()
@click.option("--in_dir", type=click.Path())
@click.option("--out_dir", type=click.Path())
@click.option("--in_sample_rate", type=int)
def resample_dataset(in_dir: str, out_dir: str, in_sample_rate: int) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    resample = ta.transforms.Resample(in_sample_rate, SAMPLING_RATE)
    for f in tqdm(Path(in_dir).glob("*.wav")):
        audio, sr = ta.load(f)
        if sr == in_sample_rate:
            ta.save(
                out_path / f.name,
                resample(audio),
                SAMPLING_RATE,
            )
        else:
            ta.save(
                out_path / f.name,
                ta.functional.resample(audio, sr, SAMPLING_RATE),
                SAMPLING_RATE,
            )


if __name__ == "__main__":
    jpmgr()
