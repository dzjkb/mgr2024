#!/usr/bin/env python
"""Explore feature thresholds and materialize a filtered subset of the sfx1 dataset.

sfx1 lives at /media/Data/jp_dir/sounds/sfx1/{train,val}/*.flac with essentia
perceptual features already extracted into sfx1_{train,val}_labels.csv (keyed by
`filename`). This script does NOT re-extract features; it reads those CSVs, applies
threshold predicates, and (in `materialize` mode) populates a new subset directory by
symlinking the passing .flac files. Training then just points at the new directory
(AudioDataset globs a flat dir), so no loader change is needed.

Run under the project env:  conda run -n mgr python build_sfx_subset.py report

Two modes:
  report      (default) print distributions + survivor counts for a grid of length
              targets x the moderate composite filter. Read-only.
  materialize given thresholds (CLI flags), symlink/copy passing files into a new
              subset dir and write manifest.csv + filters.json for provenance.

The "moderate" composite (tune with the flags below):
  - length:       dur_to_90pct_s <= <window seconds>
  - dynamic:      dynamicComplexity <= <dyn-pct> percentile
  - not-hard-tail: drop clips that are BOTH strongly pitched AND long
                   (pitched_frame_frac >= <pitch-pct> pctile AND dur_to_90pct_s
                    >= <longpitch-sec>)
  - hygiene:      drop near-silent (rms below <rms-pct> pctile) and top <outlier-pct>
                  on flatnessSFX / oddToEven
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SFX1_DIR = Path("/media/Data/jp_dir/sounds/sfx1")
SUBSET_ROOT = Path("/media/Data/jp_dir/sounds")
SAMPLING_RATE = 48000
SPLITS = ("train", "val")
# model.fixed_length must be a multiple of prod(strides)*n_bands. For sfx_v1
# (strides [4,4,4,2] -> prod 128, n_bands 16) that is 128*16 = 2048; otherwise the
# decoder floors the latent length and cannot reproduce the input (STFT-loss mismatch).
WINDOW_MULTIPLE = 2048


def snap_window_samples(length_s: float) -> int:
    """Smallest multiple of WINDOW_MULTIPLE that fits `length_s` of audio (>= it),
    so every clip kept by the `dur_to_90pct_s <= length_s` filter fits the window."""
    need = length_s * SAMPLING_RATE
    return int(-(-need // WINDOW_MULTIPLE)) * WINDOW_MULTIPLE  # ceil-div * multiple

# candidate effective-length filter thresholds (dur_to_90pct_s <= L), in seconds
LENGTH_CANDIDATES_S = [1.024, 1.6, 2.0, 3.0, 4.0]


def load_labels() -> dict[str, pd.DataFrame]:
    dfs = {}
    for split in SPLITS:
        path = SFX1_DIR / f"sfx1_{split}_labels.csv"
        df = pd.read_csv(path)
        df["__split"] = split
        dfs[split] = df
    return dfs


def combined(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs.values(), ignore_index=True)


def pct(series: pd.Series, ps=(5, 25, 50, 75, 90, 95, 99)) -> str:
    qs = np.percentile(series.values, ps)
    return "  ".join(f"p{p}={q:.3f}" for p, q in zip(ps, qs))


def keep_mask(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    length_s: float,
    dyn_pct: float,
    pitch_pct: float,
    longpitch_s: float,
    rms_pct: float,
    outlier_pct: float,
) -> pd.Series:
    """Boolean keep-mask for `df`. `length_s` is the effective-length filter
    (dur_to_90pct_s <= length_s); it is the FILTER threshold, distinct from the padded
    window recorded in provenance. Percentile thresholds are computed on `ref` (combined
    train+val) so train and val use the same cut values."""
    dyn_thr = np.percentile(ref["dynamicComplexity"].values, dyn_pct)
    pitch_thr = np.percentile(ref["pitched_frame_frac"].values, pitch_pct)
    rms_thr = np.percentile(ref["rms"].values, rms_pct)
    flat_thr = np.percentile(ref["flatnessSFX"].values, 100 - outlier_pct)
    ote_thr = np.percentile(ref["oddToEven"].values, 100 - outlier_pct)

    length_ok = df["dur_to_90pct_s"] <= length_s
    dyn_ok = df["dynamicComplexity"] <= dyn_thr
    # hard tail = strongly pitched AND long; drop it (keep the complement)
    hard_tail = (df["pitched_frame_frac"] >= pitch_thr) & (df["dur_to_90pct_s"] >= longpitch_s)
    not_hard = ~hard_tail
    not_silent = df["rms"] > rms_thr
    not_outlier = (df["flatnessSFX"] < flat_thr) & (df["oddToEven"] < ote_thr)

    return length_ok & dyn_ok & not_hard & not_silent & not_outlier


def cmd_report(args: argparse.Namespace) -> None:
    dfs = load_labels()
    all_df = combined(dfs)
    n_total = len(all_df)
    print(f"Loaded {n_total} clips ({len(dfs['train'])} train, {len(dfs['val'])} val)\n")

    print("=== Key feature distributions (combined train+val) ===")
    for col in [
        "dur_total_s", "dur_to_50pct_s", "dur_to_90pct_s",
        "dynamicComplexity", "pitched_frame_frac", "pitchConfidence_mean",
        "rms", "flatnessSFX", "oddToEven", "specFlatness_mean", "zeroCrossingRate",
    ]:
        print(f"  {col:22s} {pct(all_df[col])}")
    capped = (all_df["dur_total_s"] >= 7.99).mean()
    print(f"\n  fraction at 8s analysis cap (dur_total_s>=7.99): {capped:.1%}")
    print(f"  clips with >=1 pitched frame (n_harmonic_frames>0): "
          f"{(all_df['n_harmonic_frames'] > 0).mean():.1%}\n")

    print("=== Survivor counts: length filter x moderate composite ===")
    print(f"    (dur_to_90pct_s<=L, dyn<=p{args.dyn_pct}, drop pitched>=p{args.pitch_pct} "
          f"AND long>={args.longpitch_s}s, drop rms<p{args.rms_pct}, "
          f"drop top {args.outlier_pct}% flatnessSFX/oddToEven)")
    print(f"    window = smallest multiple of {WINDOW_MULTIPLE} that fits L\n")
    header = (f"  {'length L':>9s} {'window':>9s} {'train':>8s} {'val':>8s} "
              f"{'total':>8s} {'kept%':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for length_s in LENGTH_CANDIDATES_S:
        counts = {}
        for split, df in dfs.items():
            m = keep_mask(df, all_df, length_s, args.dyn_pct, args.pitch_pct,
                          args.longpitch_s, args.rms_pct, args.outlier_pct)
            counts[split] = int(m.sum())
        tot = counts["train"] + counts["val"]
        win = snap_window_samples(length_s)
        print(f"  {length_s:>8.3f}s {win:>9d} {counts['train']:>8d} {counts['val']:>8d} "
              f"{tot:>8d} {tot / n_total:>7.1%}")
    print("\nPick a length L + thresholds, then run `materialize --length-s L ...`.")


def cmd_materialize(args: argparse.Namespace) -> None:
    if args.length_s is None:
        raise SystemExit("materialize requires --length-s (e.g. 1.024, 1.6, 4.0)")
    dfs = load_labels()
    all_df = combined(dfs)
    window_samples = snap_window_samples(args.length_s)
    print(f"length filter dur_to_90pct_s <= {args.length_s}s  ->  "
          f"window {window_samples} samples ({window_samples / SAMPLING_RATE:.3f}s, "
          f"={window_samples // WINDOW_MULTIPLE}*{WINDOW_MULTIPLE})")

    out_root = SUBSET_ROOT / args.name
    if out_root.exists() and not args.force:
        raise SystemExit(f"{out_root} already exists; pass --force to overwrite")

    manifest_rows = []
    counts = {}
    for split, df in dfs.items():
        m = keep_mask(df, all_df, args.length_s, args.dyn_pct, args.pitch_pct,
                      args.longpitch_s, args.rms_pct, args.outlier_pct)
        kept = df[m]
        counts[split] = len(kept)
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        # clear any stale entries from a previous run so the dir matches THIS filter
        # exactly (otherwise files kept by an earlier, looser run would linger)
        for old in [*out_dir.glob("*.flac"), *out_dir.glob("*.wav")]:
            old.unlink()
        src_dir = SFX1_DIR / split
        linked = 0
        for _, row in kept.iterrows():
            fn = row["filename"]
            src = src_dir / fn
            if not src.exists():
                continue
            dst = out_dir / fn
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                import shutil
                shutil.copy2(src, dst)
            else:
                os.symlink(src.resolve(), dst)
            linked += 1
            manifest_rows.append({
                "filename": fn, "split": split,
                "dur_to_90pct_s": row["dur_to_90pct_s"],
                "dynamicComplexity": row["dynamicComplexity"],
                "pitched_frame_frac": row["pitched_frame_frac"],
                "rms": row["rms"],
            })
        print(f"  {split}: kept {len(kept)}, linked {linked} -> {out_dir}")

    pd.DataFrame(manifest_rows).to_csv(out_root / "manifest.csv", index=False)
    filters = {
        "source": str(SFX1_DIR),
        "mode": "copy" if args.copy else "symlink",
        "length_s": args.length_s,
        "window_samples": window_samples,
        "window_multiple": WINDOW_MULTIPLE,
        "dyn_pct": args.dyn_pct, "pitch_pct": args.pitch_pct,
        "longpitch_s": args.longpitch_s, "rms_pct": args.rms_pct,
        "outlier_pct": args.outlier_pct,
        "counts": counts, "total": sum(counts.values()),
    }
    (out_root / "filters.json").write_text(json.dumps(filters, indent=2))
    print(f"\nWrote {out_root}/manifest.csv and filters.json")
    print(f"Total kept: {sum(counts.values())}  window={filters['window_samples']} samples")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def add_thresholds(sp):
        sp.add_argument("--dyn-pct", type=float, default=80.0,
                        help="keep dynamicComplexity <= this percentile (default 80)")
        sp.add_argument("--pitch-pct", type=float, default=75.0,
                        help="'strongly pitched' = pitched_frame_frac >= this pctile (default 75)")
        sp.add_argument("--longpitch-s", type=float, default=1.6,
                        help="'long' threshold (s) for the pitched-AND-long hard tail (default 1.6)")
        sp.add_argument("--rms-pct", type=float, default=2.0,
                        help="drop clips with rms below this percentile (default 2)")
        sp.add_argument("--outlier-pct", type=float, default=1.0,
                        help="drop top X%% of flatnessSFX/oddToEven (default 1)")

    rp = sub.add_parser("report", help="print distributions + survivor counts (read-only)")
    add_thresholds(rp)

    mp = sub.add_parser("materialize", help="symlink/copy passing files into a new subset dir")
    add_thresholds(mp)
    mp.add_argument("--name", default="sfx1f", help="subset dir name under %s" % SUBSET_ROOT)
    mp.add_argument("--length-s", type=float, required=False,
                    help="effective-length filter: keep dur_to_90pct_s <= this (seconds). "
                         "The training window is snapped up to the next multiple of "
                         f"{WINDOW_MULTIPLE} samples and recorded in filters.json.")
    mp.add_argument("--copy", action="store_true", help="hard-copy instead of symlink")
    mp.add_argument("--force", action="store_true", help="overwrite existing subset dir")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "materialize":
        cmd_materialize(args)
    else:  # default to report
        if args.cmd is None:
            args = parser.parse_args(["report"])
        cmd_report(args)


if __name__ == "__main__":
    main()
