"""Unconditional generation for the original RAVE implementation (RAVE/).

Given a trained RAVE model and a trained (Variational) prior, autoregressively
sample latent trajectories from the prior and decode them to audio. Produces
N wav files of S seconds each in the output directory.

Example:
    conda run -n mgr python generate_unconditional.py \
        --model /media/Data/jp_dir/sounds/rave_training/vctk_words_v0_0b2e4cd2e9/version_1 \
        --prior /media/Data/jp_dir/sounds/rave_training/vctk_priors/vctk_words_fa0199570d/version_0 \
        --n 100 --seconds 1.6 --out_path outputs/uncond --gpu 0

Pass --model / --prior as either the run directory (containing config.gin and a
checkpoints/ folder) or a specific .ckpt file.
"""
import math
import os

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
import rave.prior

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, required=True,
                    help="RAVE run dir (with config.gin) or .ckpt path")
flags.DEFINE_string("prior", None, required=True,
                    help="prior run dir (with config.gin) or .ckpt path")
flags.DEFINE_integer("n", 100, help="number of samples to generate")
flags.DEFINE_float("seconds", 1.6, help="length of each sample in seconds")
flags.DEFINE_string("out_path", "outputs/uncond", help="output directory")
flags.DEFINE_integer("batch_size", 8, help="generation batch size")
flags.DEFINE_integer("gpu", -1, help="GPU index (-1 for CPU)")
flags.DEFINE_bool("argmax", False,
                  help="take argmax of the prior logits instead of sampling")
flags.DEFINE_integer("seed", None, help="random seed for reproducibility")


def load_state(model, ckpt_path):
    """Load weights, preferring EMA shadow params (as train_prior.py does)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "callbacks" in checkpoint and "EMA" in checkpoint["callbacks"]:
        model.load_state_dict(checkpoint["callbacks"]["EMA"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=False)


def main(argv):
    torch.set_float32_matmul_precision("high")
    # Autoregressive generation relies on cached convolutions (streaming buffers)
    # so each step only sees the new timestep; see Prior.generate().
    cc.use_cached_conv(True)

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda:%d" % FLAGS.gpu) if FLAGS.gpu >= 0 else torch.device("cpu")

    # --- load pretrained RAVE (the synth) ---
    synth_config = rave.core.search_for_config(FLAGS.model)
    if synth_config is None:
        raise FileNotFoundError("no config.gin found near %s" % FLAGS.model)
    gin.parse_config_file(synth_config)
    synth_run = rave.core.search_for_run(FLAGS.model)
    if synth_run is None:
        raise FileNotFoundError("no checkpoint found in %s" % FLAGS.model)
    print("RAVE model:", synth_run)
    pretrained = rave.RAVE()
    load_state(pretrained, synth_run)
    pretrained.eval()
    gin.clear_config()

    # --- load prior on top of the synth ---
    prior_config = rave.core.search_for_config(FLAGS.prior)
    if prior_config is None:
        raise FileNotFoundError("no config.gin found near %s" % FLAGS.prior)
    gin.parse_config_file(prior_config)
    prior_run = rave.core.search_for_run(FLAGS.prior)
    if prior_run is None:
        raise FileNotFoundError("no checkpoint found in %s" % FLAGS.prior)
    print("prior:", prior_run)

    if not isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        raise NotImplementedError(
            "prior generation only implemented for VariationalEncoder, got %s"
            % type(pretrained.encoder))
    prior = rave.prior.VariationalPrior(pretrained_vae=pretrained)
    load_state(prior, prior_run)
    prior.eval()

    sr = pretrained.sr

    # latents-per-second: ratio = audio samples per latent step.
    # Compute on CPU before moving to device (get_model_ratio() probes with a
    # CPU tensor internally).
    ratio = prior.get_model_ratio()
    prior = prior.to(device)

    # Number of latent steps that actually reach the decoder. The diagonal shift
    # is applied twice (forward before generate(), inverse after), and each pass
    # consumes (latent_size - 1) steps, so we seed with 2*(latent_size - 1) extra
    # steps. Mirrors Prior.validation_epoch_end, which feeds the full encoded
    # length through the same forward/inverse shift pair.
    out_latents = max(int(math.ceil(FLAGS.seconds * sr / ratio)), 1)
    seed_steps = out_latents + 2 * (prior.latent_size - 1)
    print("sr=%d, samples/latent=%d, output latent steps=%d (seed=%d, ~%.2fs)" %
          (sr, ratio, out_latents, seed_steps, out_latents * ratio / sr))

    os.makedirs(FLAGS.out_path, exist_ok=True)
    target_samples = int(round(FLAGS.seconds * sr))

    generated = 0
    pbar = tqdm.tqdm(total=FLAGS.n)
    while generated < FLAGS.n:
        bs = min(FLAGS.batch_size, FLAGS.n - generated)

        with torch.no_grad():
            # seed with a quantized sample from the standard normal prior, then
            # let generate() autoregressively overwrite every step from t=1 on.
            seed_latents = torch.randn(bs, prior.latent_size, seed_steps, device=device)
            x = prior.quantized_normal.encode(prior.diagonal_shift(seed_latents))
            x = prior.generate(x, argmax=FLAGS.argmax)

            # undo quantization + diagonal shift -> continuous latents
            z = prior.diagonal_shift.inverse(prior.quantized_normal.decode(x))
            # map latents back to audio through the pretrained RAVE decoder
            audio = prior.decode(z)  # B x C x samples

        audio = audio.cpu()
        for b in range(bs):
            wav = audio[b]
            if wav.shape[-1] >= target_samples:
                wav = wav[..., :target_samples]
            else:
                wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[-1]))
            idx = generated + b
            out_file = os.path.join(FLAGS.out_path, "sample_%05d.wav" % idx)
            torchaudio.save(out_file, wav, sample_rate=sr)

        generated += bs
        pbar.update(bs)
    pbar.close()
    print("wrote %d files to %s" % (FLAGS.n, FLAGS.out_path))


if __name__ == "__main__":
    app.run(main)
