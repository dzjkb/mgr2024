import os
import yaml
from typing import Any

import click
from cattrs import structure

from .train import do_train, TrainingConfig
from .model import ModelConfig


def _push_to_subcommands(option: str, value: str, cfg: dict[str, Any]) -> None:
    for c in cfg:
        if not isinstance(cfg[c], dict):
            continue
        cfg[c][option] = value


@click.group(context_settings={"auto_envvar_prefix": "JPMGR"})
@click.option("--config", default="./default_config.yml", type=click.Path())
@click.pass_context
def cli(ctx, config):  # type: ignore
    if os.path.exists(config):
        with open(config, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        if (dataset_dir := config.get("dataset_dir")) is not None:
            _push_to_subcommands("dataset_dir", dataset_dir, config)

        ctx.default_map = config


@cli.command()
@click.option("train_dataset_path", type=click.Path())
@click.option("val_dataset_path", type=click.Path())
@click.option("log_dir", type=click.Path())
@click.option("experiment_name", type=str)
@click.option("batch_size", type=int)
@click.option("model_config", type=click.Path())
def train(
    train_dataset_path: str,
    val_dataset_path: str,
    log_dir: str,
    experiment_name: str,
    batch_size: int,
    model_config: str,
) -> None:
    with open(model_config) as f:
        mcfg_dict = yaml.load(f, loader=yaml.SafeLoader)
        mcfg = structure(mcfg_dict, ModelConfig)
        cfg = TrainingConfig(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            log_dir=log_dir,
            experiment_name=experiment_name,
            batch_size=batch_size,
            model_config=mcfg,
        )
    do_train(cfg)


if __name__ == "__main__":
    cli()
