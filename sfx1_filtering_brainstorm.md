# Filtering `sfx1` for feasible, interesting audio modeling — brainstorm

Companion to `sfx1_eda.ipynb`. All numbers below are from the essentia feature CSVs
(`sfx1_{train,val}_labels.csv`, 57,288 clips) and reproduced by
`build_sfx_subset.py report`.

## TL;DR

- **Duration is the binding constraint, and it interacts with the thesis.** Only 45% of
  clips have ≤2s of effective sound; the median `dur_to_90pct_s` is 2.67s and **38% hit
  the 8s analysis cap**. The current `sfx/v1.yaml` cuts to 1.024s, so today the model
  trains on truncated sounds — the wrong failure mode for a "global features" latent.
- **Harmonicity and dynamic-complexity filters are gentle** — together they only remove
  ~17% of clips. They shape *character*, not size. So the interesting knob for
  feasibility is length; the interesting knob for the *kind* of sound is harmonicity.
- Because sfx1 is long-tailed, "keep ~two-thirds" and "every clip fits a short window"
  cannot both hold. We resolve this by building **two subsets** and comparing:
  - `sfx1f_short` — 1.6s window, whole sounds, **19,696 clips** (34%).
  - `sfx1f_long` — 4.0s window, keeps more of the tail whole, **27,784 clips** (48%).

## The feature vocabulary (what we can filter on)

93 essentia features per clip (see `sfx1_labels.txt`). The ones that matter for filtering:

| Axis | Columns | Reads as |
|---|---|---|
| Effective length | `dur_to_90pct_s`, `dur_to_50pct_s` | how long the sound actually is (energy-based, ignores trailing silence & the 8s cap) |
| Tonalness | `pitched_frame_frac`, `pitchConfidence_mean`, `pitchSalience_mean` | how pitched |
| Harmonic character | `inharmonicity`, `oddToEven`, `dissonance`, `tristimulus0/1/2` | *how* it's pitched (only on pitched frames) |
| Dynamics over time | `dynamicComplexity` | loudness variation → one-shot vs loop/pattern |
| Articulation | `logAttackTime`, `maxToTotal`, `TCToTotal`, `strongDecay`, `flatnessSFX` | single decaying event vs sustained/multi-hit |
| Noise↔tonal | `specFlatness_mean`, `zeroCrossingRate` | textural/noisy vs tonal |
| Hygiene | `rms`, `flatnessSFX`, `oddToEven` (tails) | silence / pathological outliers |

Prefer `dur_to_90pct_s` over `dur_total_s`: the latter is contaminated by the 8s cap
(38% of clips sit exactly at 8.0s) and by trailing silence.

## Evaluating the three proposed axes

### (a) Duration — necessary, primary, and the axis most entangled with the thesis ✅

Two very different operations hide behind "duration filtering":

- **Cut** (current behaviour): keep the clip, take the first `fixed_length` samples.
  Preserves count but **truncates the 55% of clips longer than 2s mid-sound.** For a
  latent meant to encode the *whole* global character (envelope, decay, evolution),
  training on half-sounds is directly counterproductive.
- **Filter to fit**: keep only clips whose effective sound fits the window. Every
  retained sample is seen whole. This is the right primitive for the thesis.

How much fits each length filter (`dur_to_90pct_s <= L`):

| length L | clips ≤ L | % of 57k | training window (samples) |
|---|---|---|---|
| 1.024s | 19,516 | 34.1% | 49152 (=24×2048) |
| 1.6s | 23,632 | 41.3% | 77824 (=38×2048) |
| 2.0s | 25,901 | 45.2% | 96256 (=47×2048) |
| 3.0s | 29,847 | 52.1% | 145408 (=71×2048) |
| 4.0s | 33,141 | 57.8% | 192512 (=94×2048) |

Takeaway: a short window that keeps sounds whole necessarily gives a smaller set. That
is the fundamental trade, not a tuning issue.

**Hard constraint on the window (discovered via smoke-test):** `model.fixed_length`
must be a multiple of `prod(strides) * n_bands = 128 * 16 = 2048`, not merely of
`prod(strides)=128`. The model's `do_train` assert only checks the weaker ÷128
condition, but the encoder reduction / decoder upsampling floor-divides by 2048
(`src/model.py:352`), so a non-2048 window (e.g. 76800) makes the decoder output a
different length than the input and the multi-scale STFT loss throws a shape mismatch
(observed: `tensor a (151) must match tensor b (149)`). So the "training window" column
above snaps each length up to the next multiple of 2048; the length *filter* stays at
the round value (every kept clip still fits, with a few samples of trailing pad).

### (b) Harmonicity → prefer unpitched — agree, but moderate; this is the *interestingness* knob ⚠️✅

Matches the thesis intuition (pitched resonant sustains are the last thing the model
nails). But **90.7% of clips have ≥1 confidently pitched frame** and the median
`pitched_frame_frac` is 0.33 — a hard "unpitched only" filter would gut the set and kill
timbral variety. Two refinements make it useful without being destructive:

1. **Filter on *degree*, not binary.** Drop only the strongly-tonal tail
   (`pitched_frame_frac` in its top quartile), not everything with any pitch.
2. **It's the interaction that's hard: pitched _and_ sustained.** A short pitched blip
   (a bell "ting") is easy; a long resonant drone is what the model fails on. So the
   "drop the hard tail" predicate is `pitched_frame_frac` high **AND** `dur_to_90pct_s`
   high — not pitch alone. This removes the genuinely hard cases while keeping pitched
   one-shots, which are interesting and controllable.

Left as a *reporting* dial rather than a hard filter: `specFlatness_mean` /
`zeroCrossingRate` cleanly separate noisy/textural SFX (whooshes, impacts, air — often
easier and very synth-relevant) from tonal ones, if you later want to bias the set.

### (c) Dynamic complexity → prefer simple — agree, and it doubles as a one-shot filter ✅

`dynamicComplexity` (median 6.5, p75 ≈ 14, p95 ≈ 30) is the average absolute loudness
deviation over time. High values = rhythmic / looped / multi-event material — exactly
the "temporal pattern rather than timbre" content you want to trim so the latent focuses
on timbre, and it catches loops the caption filter let through. Trimming above ~p85–p90
is low-risk and directly on-thesis. (Trimming *hard* here barely shrinks the set — most
clips are already dynamically simple — so it's cheap insurance, not a size lever.)

## Additional axes worth considering

1. **One-shot / single-event structure (strong add).** Beyond `dynamicComplexity`, the
   articulation block (`logAttackTime`, `maxToTotal`, `TCToTotal`, `strongDecay`,
   `flatnessSFX`) separates a single decaying event from a flat/sustained texture or a
   multi-hit sequence. A "front-loaded single onset" bias (peak early → low `maxToTotal`,
   `TCToTotal`) yields the synthesizer-oneshot character that made `percussive2`
   tractable and fits the "intuitive high-level control" goal. Not applied by default in
   v1 (length + dynamics already remove most multi-event clips) but available.

2. **Silence / dead-air hygiene (cheap).** `dur_total_s` p5 = 0.25s with a 0.001s min,
   and an `rms` low tail → drop near-silent/degenerate clips. Applied: drop `rms` below
   its p2.

3. **Extreme-outlier trim (hygiene, not taste).** Heavy tails: `flatnessSFX` p99 = 481
   (max ~2e8), `oddToEven` p99 = 143 (max 1000), `hfc_*`, `strongPeak_*`. Drop the top
   ~0.5–1% on `flatnessSFX`/`oddToEven` to remove pathological files that would dominate
   any unnormalised loss.

4. **Noise↔tonal balance (interestingness dial, optional).** As above — report, don't
   filter, unless you want to deliberately skew toward textural sounds.

5. **Near-duplicate pruning (future).** LAION Freesound has variation packs (same source,
   tiny edits). Not an essentia axis; a later step could cluster on MFCC-mean vectors.
   Out of scope for v1.

## The v1 recipe (the "moderate" composite actually used)

Applied as an AND; percentiles computed on the combined train+val distribution so both
splits use identical cut values (see `build_sfx_subset.py:keep_mask`):

- **Length:** `dur_to_90pct_s <= window` (1.6s for short, 4.0s for long).
- **Dynamic simplicity:** `dynamicComplexity <= p90`.
- **Not-the-hard-tail:** drop `pitched_frame_frac >= p90` **AND** `dur_to_90pct_s >= 2.5s`
  (pitched *and* long only). Pitched one-shots are kept.
- **Hygiene:** drop `rms < p2`; drop top 0.5% of `flatnessSFX` and `oddToEven`.

Resulting sets (from `build_sfx_subset.py materialize`):

| subset | length filter / window | train | val | total | % of 57k |
|---|---|---|---|---|---|
| `sfx1f_short` | ≤1.6s / 77824 | 17,761 | 1,935 | **19,696** | 34% |
| `sfx1f_long` | ≤4.0s / 192512 | 25,056 | 2,728 | **27,784** | 48% |

Note the harmonicity + dynamics + hygiene filters alone (no length cut) keep ~83%, so
the shrink is overwhelmingly the length filter. If a set feels too small, loosen the
window before loosening the character filters.

## How to regenerate / retune

```bash
conda run -n mgr python build_sfx_subset.py report            # distributions + survivor grid
conda run -n mgr python build_sfx_subset.py materialize \     # write a subset
    --name sfx1f_short --window-s 1.6 \
    --dyn-pct 90 --pitch-pct 90 --longpitch-s 2.5 --rms-pct 2 --outlier-pct 0.5
```

Each subset dir gets `manifest.csv` (kept files + the feature values used) and
`filters.json` (thresholds + counts) for provenance. Files are **symlinks** into
`sfx1/`, so subsets are cheap and disposable.

## A note on the EDA narrative

The notebook's Summary / data-quality cell says "~9% of clips are capped"; the code
itself computes **38.0%** at the 8s cap. The 9% figure looks carried over from the
percussive dataset and undersells exactly the problem this filtering addresses — worth
correcting in the notebook text.
