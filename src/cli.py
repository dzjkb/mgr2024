from typing import Any, cast
from pathlib import Path

import click
import torchaudio as ta
from cattrs import structure
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .train import do_train, TrainingConfig
from .model import SAMPLING_RATE


def parse_hydra_config(root_configs_dir: str, path: str) -> dict[str, Any]:
    GlobalHydra.instance().clear()
    abs_training_dir_path = Path(root_configs_dir).resolve()

    assert Path(path).exists(), f"experiment config {path} does not exist"
    relative_path = path.removeprefix("configs/experiment/")

    with initialize_config_dir(str(abs_training_dir_path), version_base=None):
        config = compose(config_name="default", overrides=[f"+experiment={relative_path}"])

    return cast(dict[str, Any], OmegaConf.to_object(config))


@click.group(context_settings={"auto_envvar_prefix": "JPMGR"})
def jpmgr():
    pass


@jpmgr.command()
@click.option("--config", default="configs/experiment/empty.yaml", type=click.Path())
def train(config: str) -> None:
    cfg = structure(
        parse_hydra_config("configs/", config),
        TrainingConfig,
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
