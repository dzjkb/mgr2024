from typing import Any, cast, Generator
from pathlib import Path

import click
import torchaudio as ta
from cattrs import structure
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .generate import do_generate
from .train import do_train, do_summarize,TrainingConfig
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
def jpmgr() -> None:
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
@click.option("--config", default="configs/experiment/empty.yaml", type=click.Path())
def summarize(config: str) -> None:
    cfg = structure(
        parse_hydra_config("configs/", config),
        TrainingConfig,
    )
    do_summarize(cfg)


# @jpmgr.command()
# @click.option("--checkpoint", type=click.Path())
# @click.option("--count", type=int)
# @click.option("--target_dir", type=click.Path())
# @click.option("--batch_size", type=int)
# def generate(checkpoint: str, count: int, target_dir: str, batch_size: int) -> None:
#     assert checkpoint is not None and len(checkpoint) > 0, f"checkpoint is required, got {checkpoint}"
#     do_generate(
#         checkpoint=checkpoint,
#         count=count,
#         target_dir=target_dir,
#         batch_size=batch_size
#     )


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


@jpmgr.command()
@click.option("--in_file", type=click.Path())
@click.option("--target_dir", type=click.Path())
@click.option("--target_length", type=float, help="target length of dataset samples in seconds")
@click.option("--overlap", type=float, help="overlap of dataset samples in seconds")
def dataset_from_file(in_file: str, target_dir: str, target_length: float, overlap: float) -> None:
    """
    this command cuts a given audio file into clips of `target_length` seconds and puts them in `target_dir`.
    """

    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True, parents=True)

    target_length_samples = int(target_length * SAMPLING_RATE)
    overlap_samples = int(overlap * SAMPLING_RATE)

    def _overlapping_chunks(
        total_length: int, chunk_length: int, chunk_overlap: int
    ) -> Generator[tuple[int, int], None, None]:
        for i in range(0, total_length, chunk_length - chunk_overlap):
            yield i, i+chunk_length

    in_audio, sr = ta.load(in_file)
    if sr != SAMPLING_RATE:
        in_audio = ta.functional.resample(in_audio, sr, SAMPLING_RATE)

    in_name = Path(in_file).stem
    for idx, (start, end) in tqdm(
        enumerate(
            _overlapping_chunks(in_audio.shape[-1], target_length_samples, overlap_samples)
        ),
        total=in_audio.shape[-1] // (target_length_samples - overlap_samples),
    ):
        ta.save(
            target_path / f"{in_name}_part_{idx}.wav",
            in_audio[..., start:end],
            SAMPLING_RATE,
        )


if __name__ == "__main__":
    jpmgr()
