#!/bin/bash
# Run the full eval suite (reconstruction + unconditional generation) on the
# SFX validation set for the no-NF, NF, NF+prior and RAVE models.
#
# We copy the val set into a fresh per-run dir because several commands write
# new directories *next to* their inputs (reconstruction AudioTensors, CLAP
# embedding caches like `<dir>_embeddings`). Copying isolates each run so
# concurrent/repeated runs don't clobber each other. The val set is small.
#
# Environments: the thesis CLI (`src.cli`) runs in the `mgr` conda env; the
# original RAVE scripts (reconstruct.py / generate_unconditional.py) need
# `mgr-rave`. We activate `mgr` for the whole run and hop into `mgr-rave` only
# around the RAVE steps. `conda activate` (not `conda run`) is required so the
# RAVE scripts' absl --flags parse correctly.
#
# All console output is tee'd to <run_dir>/run.log, and the metric lines are
# grepped into <run_dir>/results.txt at the end for easy reading.

set -euo pipefail

# Make `conda activate` usable in a non-interactive shell.
source /home/jp/miniconda3/etc/profile.d/conda.sh
conda activate mgr

export CLAP_DEVICE="cuda:0"
DEVICE="cuda:0"
RAVE_GPU=0
DATA_LENGTH="1.621"
GEN_COUNT=1000

EVALS_ROOT_DIR="/media/Data/jp_dir/sounds/specific_eval_dirs"
SFX_VAL_SET="/media/Data/jp_dir/sounds/sfx1f_short/val"

NONF_CONFIG="configs/experiment/sfx/v1_short.yaml"
NONF_CKPT="logs/sfx1f_short/version_0/checkpoints/last.ckpt"

NF_CONFIG="configs/experiment/sfx/v1_short_nf.yaml"
NF_CKPT="logs/sfx1f_short_nf/version_0/checkpoints/last.ckpt"

NF_PRIOR_CONFIG="prior_configs/v3.yaml"
NF_PRIOR_CKPT="logs/sfx_short_nf_v0_prior/version_0/checkpoints/epoch=199-step=1600.ckpt"

RAVE_MODEL="/media/Data/jp_dir/sounds/rave_training/sfx_short_0b2e4cd2e9/version_0"
RAVE_PRIOR="/media/Data/jp_dir/sounds/rave_training/sfx_priors/sfx_short_7de6d3ab59/version_0"

# Hop conda envs. `conda deactivate` then activate the target so the RAVE
# scripts see exactly mgr-rave (and vice versa) rather than a stacked env.
use_rave_env () { conda deactivate && conda activate mgr-rave; }
use_mgr_env  () { conda deactivate && conda activate mgr; }

# === prepare eval dir ===

# Timestamped run dir so repeated runs never collide. The copied val set is the
# reference set for every metric in this run (its `_embeddings` cache lands here
# too, not next to the original val set).
RUN_DIR="${EVALS_ROOT_DIR}/sfx_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

REF_SET="${RUN_DIR}/val_set"
echo "[prep] copying val set -> ${REF_SET}"
cp -r "$SFX_VAL_SET" "$REF_SET"

LOG_FILE="${RUN_DIR}/run.log"
RESULTS_FILE="${RUN_DIR}/results.txt"
: > "$RESULTS_FILE"  # evaluate_dir appends to this as each eval completes
# tee everything from here on to the run log.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[prep] run dir: ${RUN_DIR}"

# Evaluate a wav directory against the reference (copied) val set. The metric
# line is appended to RESULTS_FILE as it is produced (not grepped from the log
# afterwards): `exec > >(tee ...)` runs tee asynchronously and bash never reaps
# it, so a post-hoc grep of LOG_FILE can race the final flush and drop the last
# result. Writing here makes the summary deterministic.
evaluate_dir () {
    local label="$1" target_dir="$2"
    echo "== ${label}" >> "$RESULTS_FILE"
    python -m src.cli evaluate-embedding-dir \
        --reference_set "$REF_SET" \
        --target_dir "$target_dir" \
        --data_length "$DATA_LENGTH" \
        | tee -a "$RESULTS_FILE"
}

# === reconstructions ===
# do_reconstruct writes an AudioTensor to `<source>_reconstructions_<suffix>`;
# we then convert that to a wav dir we can embed and evaluate.

run_reconstruction () {
    local name="$1" config="$2" ckpt="$3"
    local at_suffix="${name}_audiotensor"
    local at_dir="${REF_SET}_reconstructions_${at_suffix}"
    local wav_dir="${REF_SET}_reconstructions_${name}"

    echo "[recon:${name}] reconstructing into ${wav_dir}"
    python -m src.cli reconstruct \
        --config_path "$config" \
        --checkpoint "$ckpt" \
        --source_dir "$REF_SET" \
        --device "$DEVICE" \
        --suffix "$at_suffix"

    python -m src.cli convert-audio-tensor-to-wavs \
        --source_dir "$at_dir" \
        --target_dir "$wav_dir" \
        --sample_rate 48000

    echo "[recon:${name}] evaluating ${wav_dir}"
    evaluate_dir "reconstruction:${name}" "$wav_dir"
}

run_reconstruction "nonf" "$NONF_CONFIG" "$NONF_CKPT"
run_reconstruction "nf"   "$NF_CONFIG"   "$NF_CKPT"

# RAVE reconstruction - runs in `mgr-rave`. Reconstruct the *copied* val set
# into a dir inside this run, then hop back to `mgr` to evaluate it.
RAVE_RECON_DIR="${RUN_DIR}/recon_rave"
echo "[recon:rave] reconstructing into ${RAVE_RECON_DIR} (mgr-rave env)"
use_rave_env
python /home/jp/mgr2024/reconstruct.py \
    --model "$RAVE_MODEL" \
    --in_dir "$REF_SET" \
    --out_dir "$RAVE_RECON_DIR" \
    --gpu "$RAVE_GPU" --batch_size 32
use_mgr_env
evaluate_dir "reconstruction:rave" "$RAVE_RECON_DIR"

# === unconditional generation ===

GEN_NONF_DIR="${RUN_DIR}/gen_nonf"
GEN_NF_DIR="${RUN_DIR}/gen_nf"
GEN_NF_PRIOR_DIR="${RUN_DIR}/gen_nf_prior"
GEN_RAVE_DIR="${RUN_DIR}/gen_rave"

echo "[gen:nonf] generating ${GEN_COUNT} into ${GEN_NONF_DIR}"
python -m src.cli generate \
    --config_path "$NONF_CONFIG" \
    --checkpoint "$NONF_CKPT" \
    --count "$GEN_COUNT" \
    --target_dir "$GEN_NONF_DIR" \
    --device "$DEVICE"

echo "[gen:nf] generating ${GEN_COUNT} into ${GEN_NF_DIR}"
python -m src.cli generate \
    --config_path "$NF_CONFIG" \
    --checkpoint "$NF_CKPT" \
    --count "$GEN_COUNT" \
    --target_dir "$GEN_NF_DIR" \
    --device "$DEVICE"

echo "[gen:nf_prior] generating ${GEN_COUNT} into ${GEN_NF_PRIOR_DIR}"
python -m src.cli generate \
    --config_path "$NF_CONFIG" \
    --checkpoint "$NF_CKPT" \
    --count "$GEN_COUNT" \
    --target_dir "$GEN_NF_PRIOR_DIR" \
    --device "$DEVICE" \
    --prior_config_path "$NF_PRIOR_CONFIG" \
    --prior_checkpoint "$NF_PRIOR_CKPT"

# RAVE prior generation - runs in `mgr-rave`, then hop back to `mgr` to eval.
echo "[gen:rave] generating ${GEN_COUNT} into ${GEN_RAVE_DIR} (mgr-rave env)"
use_rave_env
python /home/jp/mgr2024/generate_unconditional.py \
    --model "$RAVE_MODEL" \
    --prior "$RAVE_PRIOR" \
    --n "$GEN_COUNT" --seconds "$DATA_LENGTH" \
    --out_path "$GEN_RAVE_DIR" \
    --batch_size 16 --gpu "$RAVE_GPU" --seed 0
use_mgr_env

evaluate_dir "generation:nonf"     "$GEN_NONF_DIR"
evaluate_dir "generation:nf"       "$GEN_NF_DIR"
evaluate_dir "generation:nf_prior" "$GEN_NF_PRIOR_DIR"
evaluate_dir "generation:rave"     "$GEN_RAVE_DIR"

# === results ===
# RESULTS_FILE was built incrementally by evaluate_dir (one "== <label>" line
# per eval, followed by its KID/FAD/Vendi line). Just show it.
echo
echo "===== SUMMARY (${RESULTS_FILE}) ====="
cat "$RESULTS_FILE"
echo "====================================="
