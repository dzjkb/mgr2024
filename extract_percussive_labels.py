#!/usr/bin/env python
"""Extract essentia perceptual labels for percussive one-shot samples.

Run under the `essentia` conda env (has essentia but NOT soundfile/torch):

    conda run -n essentia python extract_percussive_labels.py \
        --in_dir /media/Data/jp_dir/sounds/percussive2_val \
        --out_csv percussive2_val_labels.csv

Produces one row per audio file, keyed by `filename` (basename, matching the
key used in the latent dump's metadata.json), with:

  * instrument `family`  -- the ONLY label parsed from the filename
  * continuous perceptual axes from essentia (feature groups A-E), all kept
    continuous (no binning) so they can be used as regression targets against
    the 256-d latent.

Framewise features are pooled over the clip (mean + std, + max where an onset
peak is meaningful) so every row is fixed-length regardless of clip duration.
Long outliers (dataset max ~86s) are handled by capping the analysed length.
Harmonic descriptors are gated on pitch confidence, and the spectral-peak list
is guarded (HarmonicPeaks rejects a 0 Hz peak), so noisy samples don't crash.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import essentia.standard as es

# --- audio / analysis constants -------------------------------------------
SR = 48000            # match the model's SAMPLING_RATE
FRAME = 2048
HOP = 1024
MAX_SECONDS = 8.0     # cap analysis length; guards the ~86s tail outliers
MAX_SAMPLES = int(MAX_SECONDS * SR)
PITCH_CONF_GATE = 0.5  # harmonic descriptors only trusted above this
AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")


# --- instrument family (only filename-derived label) ----------------------
# Ordered: first matching pattern wins. Applied to the lowercased basename.
FAMILY_PATTERNS = [
    ("kick",    r"\b(kick|vkick|bd|bassdrum|bass_?drum|808)\b|kick|vkickp"),
    ("snare",   r"snare|snr|rimshot|rim"),
    ("hat",     r"hat|hi-?hat|hh\b|openhat|closedhat"),
    ("cymbal",  r"crash|ride|cymbal|splash|china"),
    ("tom",     r"\btom\b|tom(_|-|\d)"),
    ("clap",    r"clap|handclap"),
    ("perc",    r"perc|conga|bongo|shaker|tamb|cowbell|clave|block|djembe|cabasa|guiro|tabla"),
]
_HASH_RE = re.compile(r"^[0-9a-f]{6,10}$")


def parse_family(basename: str) -> str:
    stem = os.path.splitext(basename)[0]
    low = stem.lower()
    # Opaque hash names carry no instrument info.
    if _HASH_RE.match(low):
        return "unknown"
    for fam, pat in FAMILY_PATTERNS:
        if re.search(pat, low):
            return fam
    return "unknown"


# --- helpers --------------------------------------------------------------
def _stats(prefix, arr):
    """mean/std over a framewise scalar list."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
    return {f"{prefix}_mean": float(a.mean()), f"{prefix}_std": float(a.std())}


def _vec_stats(prefix, mat):
    """mean/std over a framewise vector list -> per-dim columns."""
    m = np.asarray(mat, dtype=np.float64)
    out = {}
    if m.ndim != 2 or m.shape[0] == 0:
        return out
    mean = m.mean(0)
    std = m.std(0)
    for i in range(m.shape[1]):
        out[f"{prefix}{i}_mean"] = float(mean[i])
        out[f"{prefix}{i}_std"] = float(std[i])
    return out


# --- per-file feature extraction ------------------------------------------
def extract(filename: str) -> dict:
    audio = es.MonoLoader(filename=filename, sampleRate=SR)()
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    row = {}
    n = len(audio)
    if n < FRAME:
        # too short to frame reliably; pad for the framewise pass
        audio_padded = np.pad(audio, (0, FRAME - n))
    else:
        audio_padded = audio

    # ---- A. Temporal envelope / articulation (whole-signal) --------------
    env = es.Envelope()(audio)
    row["logAttackTime"] = float(es.LogAttackTime()(env)[0])
    row["TCToTotal"] = float(es.TCToTotal()(env))
    row["maxToTotal"] = float(es.MaxToTotal()(env))
    row["flatnessSFX"] = float(es.FlatnessSFX()(env))
    try:
        row["strongDecay"] = float(es.StrongDecay()(audio))
    except RuntimeError:
        row["strongDecay"] = 0.0
    # effective (RMS-energy) duration: fraction/length where cumulative energy
    # crosses thresholds -> proxy for short/regular/deep without the filename.
    e = audio.astype(np.float64) ** 2
    tot = e.sum()
    if tot > 0:
        cdf = np.cumsum(e) / tot
        row["dur_total_s"] = n / SR
        row["dur_to_50pct_s"] = int(np.searchsorted(cdf, 0.5)) / SR
        row["dur_to_90pct_s"] = int(np.searchsorted(cdf, 0.9)) / SR
    else:
        row["dur_total_s"] = n / SR
        row["dur_to_50pct_s"] = 0.0
        row["dur_to_90pct_s"] = 0.0

    # ---- E. Loudness / dynamics (whole-signal) ---------------------------
    row["rms"] = float(es.RMS()(audio))
    row["loudness"] = float(es.Loudness()(audio))
    try:
        row["loudnessVickers"] = float(es.LoudnessVickers()(audio))
    except RuntimeError:
        row["loudnessVickers"] = 0.0
    try:
        row["dynamicComplexity"] = float(es.DynamicComplexity(sampleRate=SR)(audio)[0])
    except RuntimeError:
        row["dynamicComplexity"] = 0.0
    row["zeroCrossingRate"] = float(es.ZeroCrossingRate()(audio))

    # ---- framewise algos (B brightness, C timbre, D pitch/harmonic) ------
    w = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    centroid = es.Centroid(range=SR / 2)
    rolloff = es.RollOff(sampleRate=SR)
    flatness = es.Flatness()
    decrease = es.Decrease(range=SR / 2)
    crest = es.Crest()
    spec_complexity = es.SpectralComplexity(sampleRate=SR)
    strong_peak = es.StrongPeak()
    hfc = es.HFC(sampleRate=SR)
    pitch_salience = es.PitchSalience(sampleRate=SR)
    pyin = es.PitchYinFFT(frameSize=FRAME, sampleRate=SR)
    spec_contrast = es.SpectralContrast(frameSize=FRAME, sampleRate=SR)
    mfcc = es.MFCC(inputSize=FRAME // 2 + 1, sampleRate=SR)
    peaks = es.SpectralPeaks(sampleRate=SR, minFrequency=1.0)
    harm_peaks = es.HarmonicPeaks()
    inharm = es.Inharmonicity()
    trist = es.Tristimulus()
    o2e = es.OddToEvenHarmonicEnergyRatio()
    dissonance = es.Dissonance()

    B_centroid, B_rolloff, B_flatness, B_decrease = [], [], [], []
    B_crest, B_complexity, B_strongpeak, B_hfc = [], [], [], []
    C_mfcc, C_contrast, C_valley = [], [], []
    D_pitch, D_conf, D_salience = [], [], []
    D_inharm, D_o2e, D_diss = [], [], []
    D_trist = []

    for frame in es.FrameGenerator(audio_padded, frameSize=FRAME, hopSize=HOP,
                                   startFromZero=True):
        s = spectrum(w(frame))
        if s.sum() <= 0:
            continue
        # B. brightness / spectral shape
        B_centroid.append(centroid(s))
        B_rolloff.append(rolloff(s))
        B_flatness.append(flatness(s))
        B_decrease.append(decrease(s))
        B_crest.append(crest(s))
        B_complexity.append(spec_complexity(s))
        B_strongpeak.append(strong_peak(s))
        B_hfc.append(hfc(s))
        # C. timbre fingerprint
        _, mfc = mfcc(s)
        C_mfcc.append(mfc)
        c, v = spec_contrast(s)
        C_contrast.append(c)
        C_valley.append(v)
        # D. pitch / tonalness
        pitch, conf = pyin(s)
        D_pitch.append(pitch)
        D_conf.append(conf)
        D_salience.append(pitch_salience(s))
        # harmonic descriptors: only when this frame is confidently pitched
        # and the peak list is well-formed (HarmonicPeaks rejects 0 Hz peaks).
        if conf >= PITCH_CONF_GATE and pitch > 0:
            pfreq, pmag = peaks(s)
            if len(pfreq) >= 2 and pfreq[0] > 0:
                try:
                    hfreq, hmag = harm_peaks(pfreq, pmag, float(pitch))
                    if len(hfreq) >= 2 and hfreq[0] > 0:
                        D_inharm.append(inharm(hfreq, hmag))
                        D_o2e.append(o2e(hfreq, hmag))
                        D_trist.append(trist(hfreq, hmag))
                    D_diss.append(dissonance(pfreq, pmag))
                except RuntimeError:
                    pass

    # ---- pool framewise -> row -------------------------------------------
    row.update(_stats("centroid", B_centroid))
    row.update(_stats("rolloff", B_rolloff))
    row.update(_stats("specFlatness", B_flatness))
    row.update(_stats("specDecrease", B_decrease))
    row.update(_stats("specCrest", B_crest))
    row.update(_stats("specComplexity", B_complexity))
    row.update(_stats("strongPeak", B_strongpeak))
    row.update(_stats("hfc", B_hfc))

    row.update(_vec_stats("mfcc", C_mfcc))
    row.update(_vec_stats("contrast", C_contrast))
    row.update(_vec_stats("valley", C_valley))

    row.update(_stats("pitch", D_pitch))
    row.update(_stats("pitchConfidence", D_conf))
    row.update(_stats("pitchSalience", D_salience))
    # tonalness summary: fraction of frames confidently pitched
    conf_arr = np.asarray(D_conf, dtype=np.float64)
    row["pitched_frame_frac"] = float((conf_arr >= PITCH_CONF_GATE).mean()) if conf_arr.size else 0.0
    row["n_harmonic_frames"] = len(D_inharm)
    # harmonic descriptors: mean over gated frames (0 if never pitched)
    row["inharmonicity"] = float(np.mean(D_inharm)) if D_inharm else 0.0
    row["oddToEven"] = float(np.mean(D_o2e)) if D_o2e else 0.0
    row["dissonance"] = float(np.mean(D_diss)) if D_diss else 0.0
    if D_trist:
        tmean = np.mean(np.asarray(D_trist), axis=0)
        for i in range(len(tmean)):
            row[f"tristimulus{i}"] = float(tmean[i])
    else:
        for i in range(3):
            row[f"tristimulus{i}"] = 0.0

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="directory of audio one-shots")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N files (0 = all); for smoke tests")
    ap.add_argument("--no_family", action="store_true",
                    help="omit the filename-derived instrument `family` column "
                         "(e.g. for non-percussive datasets where it doesn't apply)")
    args = ap.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(args.in_dir, "*"))
        if f.lower().endswith(AUDIO_EXTS)
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No audio files found in {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting features for {len(files)} files from {args.in_dir} ...",
          file=sys.stderr)

    rows = []
    fieldset = set()
    failures = []
    for i, f in enumerate(files, 1):
        base = os.path.basename(f)
        try:
            feats = extract(f)
        except Exception as e:  # never let one bad file kill the whole run
            failures.append((base, repr(e)))
            print(f"  [{i}/{len(files)}] FAILED {base}: {e!r}", file=sys.stderr)
            continue
        row = {"filename": base}
        if not args.no_family:
            row["family"] = parse_family(base)
        row.update(feats)
        rows.append(row)
        fieldset.update(row.keys())
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {base}", file=sys.stderr)

    # stable column order: keys first, then sorted feature columns
    lead = ["filename"] if args.no_family else ["filename", "family"]
    feat_cols = sorted(c for c in fieldset if c not in lead)
    fieldnames = lead + feat_cols

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, 0.0) for k in fieldnames})

    print(f"\nWrote {len(rows)} rows x {len(fieldnames)} cols -> {args.out_csv}",
          file=sys.stderr)
    if failures:
        print(f"{len(failures)} file(s) failed:", file=sys.stderr)
        for b, e in failures:
            print(f"  {b}: {e}", file=sys.stderr)
    # quick family breakdown
    if not args.no_family:
        from collections import Counter
        fam = Counter(r["family"] for r in rows)
        print("family counts:", dict(fam.most_common()), file=sys.stderr)


if __name__ == "__main__":
    main()
