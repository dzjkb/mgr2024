import os
import gc
import sys
import ctypes
import ctypes.util
from collections import defaultdict

import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
from attrs import frozen, asdict
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from .model import VAE, ModelConfig
from .dataset import AudioDataset, DatasetConfig
from .callbacks import init_callbacks, CallbacksConfig
from .noise import NoiseConfig

_worker_batch_counter = 0


def trim_memory() -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    return libc.malloc_trim(0) 


def print_worker_memory_report(top_n=10):
    """
    Comprehensive memory report covering:
    1. gc-visible Python objects (by total shallow size)
    2. UntypedStorage / tensor buffers (actual data size via nbytes)
    3. Process RSS from /proc — the ground truth for what the OS sees
    """
    pid = os.getpid()
    worker_info = None
    try:
        import torch
        worker_info = torch.utils.data.get_worker_info()
    except Exception:
        pass
    label = f"Worker {worker_info.id}" if worker_info else "Main"

    # ------------------------------------------------------------------
    # 1. RSS from /proc — this is the ground truth, includes everything:
    #    Python heap, torch C buffers, memory-mapped files, shared memory
    # ------------------------------------------------------------------
    rss_bytes = None
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024  # kB -> bytes
                    break
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. UntypedStorage actual buffer sizes (what gc.get_objects misses)
    # ------------------------------------------------------------------
    storage_total = 0
    storage_count = 0
    tensor_total = 0
    tensor_count = 0

    try:
        import torch
        for obj in gc.get_objects():
            if isinstance(obj, torch.UntypedStorage):
                try:
                    storage_total += obj.nbytes()  # actual buffer, not struct
                    storage_count += 1
                except Exception:
                    pass
            elif isinstance(obj, torch.Tensor):
                try:
                    if obj.device.type == "cpu":
                        tensor_total += obj.nbytes()
                        tensor_count += 1
                except Exception:
                    pass
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 3. Standard gc shallow scan for everything else
    # ------------------------------------------------------------------
    type_sizes = defaultdict(lambda: [0, 0])
    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            name = type(obj).__name__
            type_sizes[name][0] += size
            type_sizes[name][1] += 1
        except TypeError:
            pass
    sorted_types = sorted(type_sizes.items(), key=lambda x: x[1][0], reverse=True)

    def fmt_bytes(n):
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[{label} | PID {pid}]")
    if rss_bytes is not None:
        print(f"  Process RSS (OS ground truth) : {fmt_bytes(rss_bytes)}")
    print(f"  UntypedStorage buffers (real) : {fmt_bytes(storage_total)} across {storage_count} storages")
    print(f"  CPU Tensors (real nbytes)     : {fmt_bytes(tensor_total)} across {tensor_count} tensors")
    print(f"  gc-visible Python heap        : {fmt_bytes(sum(v[0] for v in type_sizes.values()))}")
    print(f"\n  Top {top_n} Python types by shallow size:")
    print(f"  {'Rank':<6} {'Type':<40} {'Total Size':>12} {'Count':>10}")
    print(f"  {'-' * 66}")
    for rank, (type_name, (total_bytes, count)) in enumerate(sorted_types[:top_n], 1):
        print(f"  {rank:<6} {type_name:<40} {fmt_bytes(total_bytes):>12} {count:>10,}")
    print(f"{'='*70}\n", flush=True)


def diagnosing_collate_fn(batch):
    """
    Wraps your existing collate, and periodically prints the memory report
    from inside the worker process.
    """
    import torch  # import here to ensure worker-local reference
    global _worker_batch_counter

    # --- your real collate logic here (or just use default_collate) -------
    collated = torch.utils.data.default_collate(batch)
    # ----------------------------------------------------------------------

    _worker_batch_counter += 1
    if _worker_batch_counter % 20 == 0:
        # print_worker_memory_report()
        trim_memory()

    return collated


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
    val_every: int
    model: ModelConfig
    callbacks: CallbacksConfig
    dataset: DatasetConfig
    noise: NoiseConfig | None = None
    test_set_path: str | None = None


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
    assert cfg.model.mono == cfg.dataset.mono, f"both model and dataset must have the same mono setting, got model: {cfg.model.mono}, dataset: {cfg.dataset.mono}"

    torch.manual_seed(420)
    model = VAE(noise_config=cfg.noise, **asdict(cfg.model), test_set_path=cfg.test_set_path)
    train_set = DataLoader(
        AudioDataset(Path(cfg.train_dataset_path), **asdict(cfg.dataset)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=diagnosing_collate_fn,
    )
    val_set = DataLoader(
        AudioDataset(Path(cfg.val_dataset_path), **asdict(cfg.dataset.val_overrides())),
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
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        devices=1,
        callbacks=init_callbacks(cfg.callbacks),
        profiler="pytorch",
        enable_progress_bar=True,
        check_val_every_n_epoch=cfg.val_every,
    )
    checkpoint_kwarg: dict[str, str] = (
        {"ckpt_path": cfg.checkpoint_path, "weights_only": False} if cfg.checkpoint_path is not None else {}
    )
    trainer.fit(model, train_set, val_set, **checkpoint_kwarg)  # type: ignore


def do_summarize(cfg: TrainingConfig) -> None:
    model = VAE(noise_config=cfg.noise, **asdict(cfg.model))
    print(summarize(model))
