from typing import Generator
from pathlib import Path

import click
import torchaudio as ta
from cattrs import structure
from tqdm import tqdm

from .generate import do_generate
from .train import do_train, do_summarize,TrainingConfig
from .model import SAMPLING_RATE
from .evaluations.kid import kid_for_audio_directories, kid_for_serialized_tensors, kid_for_serialized_embeddings
from .ds_utils import embed_directory as do_embed_directory
from .ds_utils import save_audio_tensor
from .configs import parse_hydra_config


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


@jpmgr.command()
@click.option("--config_path", type=click.Path())
@click.option("--checkpoint", type=click.Path())
@click.option("--count", type=int)
@click.option("--target_dir", type=click.Path())
@click.option("--batch_size", type=int | None)
@click.option("--device", type=str | None)
def generate(
    config_path: str,
    checkpoint: str,
    count: int,
    target_dir: str,
    batch_size: int | None,
    device: str | None,
) -> None:
    assert checkpoint is not None and len(checkpoint) > 0, f"checkpoint is required, got {checkpoint}"
    do_generate(
        config_path=config_path,
        checkpoint=checkpoint,
        count=count,
        target_dir=target_dir,
        batch_size=batch_size,
        device=device,
    )


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


@jpmgr.command()
@click.option("--reference_set", type=click.Path())
@click.option("--target_set", type=click.Path())
@click.option("--format", type=str)
@click.option("--data_length", type=float)
def calculate_kid(reference_set: str, target_set: str, format: str, data_length: float) -> None:
    match format:
        case "directory":
            assert data_length is not None, "data_length is required if format == 'directory'"
            kid = kid_for_audio_directories(reference_set, target_set, data_length)
        case "serialized":
            kid = kid_for_serialized_tensors(reference_set, target_set)
        case "embeddings":
            kid = kid_for_serialized_embeddings(reference_set, target_set)
        case _:
            assert False, "expected format to be one of: (directory, serialized)"

    print("==========================")
    print(f"kid = {kid:.8f}")
    print("==========================")


@jpmgr.command()
@click.option("--reference_set", type=click.Path())
@click.option("--target_set", type=click.Path())
@click.option("--format", type=str)
@click.option("--data_length", type=float)
def calculate_fad(reference_set: str, target_set: str, format: str, data_length: float) -> None:
    # TODO
    # match format:
    #     case "directory":
    #         assert data_length is not None, "data_length is required if format == 'directory'"
    #         kid = kid_for_audio_directories(reference_set, target_set, data_length)
    #     case "serialized":
    #         kid = kid_for_serialized_tensors(reference_set, target_set)
    #     case "embeddings":
    #         kid = kid_for_serialized_embeddings(reference_set, target_set)
    #     case _:
    #         assert False, "expected format to be one of: (directory, serialized)"

    # print("==========================")
    # print(f"fad = {fad:.8f}")
    # print("==========================")
    pass


@jpmgr.command()
@click.option("--audio_path", type=click.Path())
@click.option("--target_path", type=click.Path())
@click.option("--data_length", type=float)
def embed_directory(audio_path: str, target_path: str, data_length: float) -> None:
    embeddings = do_embed_directory(audio_path, data_length)
    save_audio_tensor(embeddings, target_path)


if __name__ == "__main__":
    jpmgr()
