"""Reconstruct audio files through a trained RAVE model (encode -> decode).

Loads only the RAVE model (no prior), runs each input file through
model.forward() and writes the reconstruction under the same filename in the
target dir. Files are bucketed by length so a batch contains equal-length
clips; output is trimmed back to each input's exact length.

Multi-GPU: pass several --gpu flags (e.g. --gpu 0 --gpu 1) and launch one
process per GPU with --shard_index / --shard_count to split the file list.

Example (single GPU):
    conda activate mgr-rave
    python reconstruct.py \
        --model /media/.../vctk_words_v0_0b2e4cd2e9/version_1 \
        --in_dir  /media/.../vctk_words_v2/audio_val \
        --out_dir /media/.../vctk_words_v2/audio_val_ravev1_reconstructed \
        --gpu 0 --batch_size 16
"""
import os
from collections import defaultdict

import gin
import torch
import torchaudio
import tqdm
from absl import app, flags

import cached_conv as cc

try:
    import rave
except ImportError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "RAVE")))
    import rave

import rave.core

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, required=True,
                    help="RAVE run dir (with config.gin) or .ckpt path")
flags.DEFINE_string("in_dir", None, required=True, help="input audio directory")
flags.DEFINE_string("out_dir", None, required=True, help="output directory")
flags.DEFINE_integer("gpu", -1, help="GPU index (-1 for CPU)")
flags.DEFINE_integer("batch_size", 16, help="reconstruction batch size")
flags.DEFINE_integer("shard_index", 0, help="this process's shard (0-based)")
flags.DEFINE_integer("shard_count", 1, help="total number of shards/processes")
flags.DEFINE_bool("skip_existing", True, help="skip files already present in out_dir")


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".mp3"}


def list_audio_files(in_dir):
    files = [f for f in os.listdir(in_dir)
             if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    return sorted(files)


def main(argv):
    torch.set_float32_matmul_precision("high")
    # non-streaming: standard (non-cached) convs, whole clip in one forward pass
    cc.use_cached_conv(False)

    device = torch.device("cuda:%d" % FLAGS.gpu) if FLAGS.gpu >= 0 else torch.device("cpu")

    config = rave.core.search_for_config(FLAGS.model)
    if config is None:
        raise FileNotFoundError("no config.gin found near %s" % FLAGS.model)
    gin.parse_config_file(config)
    run = rave.core.search_for_run(FLAGS.model)
    if run is None:
        raise FileNotFoundError("no checkpoint found in %s" % FLAGS.model)
    print("[shard %d/%d] RAVE model: %s" % (FLAGS.shard_index, FLAGS.shard_count, run))
    model = rave.RAVE()
    checkpoint = torch.load(run, map_location="cpu")
    if "callbacks" in checkpoint and "EMA" in checkpoint["callbacks"]:
        model.load_state_dict(checkpoint["callbacks"]["EMA"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model = model.to(device)
    sr = model.sr

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    files = list_audio_files(FLAGS.in_dir)
    # deterministic shard split across processes/GPUs
    files = files[FLAGS.shard_index::FLAGS.shard_count]
    if FLAGS.skip_existing:
        files = [f for f in files
                 if not os.path.exists(os.path.join(FLAGS.out_dir, f))]
    print("[shard %d/%d] %d files to process" %
          (FLAGS.shard_index, FLAGS.shard_count, len(files)))

    # bucket by sample length so each batch is rectangular (no padding artifacts)
    by_len = defaultdict(list)
    for f in files:
        info = torchaudio.info(os.path.join(FLAGS.in_dir, f))
        by_len[info.num_frames].append(f)

    pbar = tqdm.tqdm(total=len(files), position=FLAGS.shard_index,
                     desc="gpu%d" % FLAGS.gpu)
    for length, group in by_len.items():
        for i in range(0, len(group), FLAGS.batch_size):
            names = group[i:i + FLAGS.batch_size]
            wavs = []
            for name in names:
                x, file_sr = torchaudio.load(os.path.join(FLAGS.in_dir, name))
                if file_sr != sr:
                    x = torchaudio.functional.resample(x, file_sr, sr)
                if x.shape[0] != model.n_channels:
                    x = x[:model.n_channels] if x.shape[0] > model.n_channels \
                        else x.expand(model.n_channels, -1)
                wavs.append(x)
            batch = torch.stack(wavs, 0).to(device)  # B x C x T

            with torch.no_grad():
                out = model.forward(batch)
            out = out[..., :batch.shape[-1]].cpu()  # trim to input length

            for name, y in zip(names, out):
                torchaudio.save(os.path.join(FLAGS.out_dir, name), y, sample_rate=sr)
            pbar.update(len(names))
    pbar.close()
    print("[shard %d/%d] done" % (FLAGS.shard_index, FLAGS.shard_count))


if __name__ == "__main__":
    app.run(main)
