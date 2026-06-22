import numpy as np
import torch
from fadtk.fad import FADInfResults, calc_embd_statistics, calc_frechet_distance
from rich.progress import Progress, TextColumn

from ..ds_utils import load_audio_tensor


def _as_embedding_array(x) -> np.ndarray:
    """Accept torch.Tensor / np.ndarray / AudioTensor and return a 2D (n, d) float array."""
    if hasattr(x, "data") and not isinstance(x, (np.ndarray, torch.Tensor)):
        # duck-type AudioTensor (from src.ds_utils)
        x = x.data
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert x.ndim == 2, f"expected 2D (n_samples, n_features) embeddings, got shape {x.shape}"
    return x.astype(np.float64, copy=False)


def fad_from_embeddings(
    reference_embeddings,
    target_embeddings,
    steps: int = 50,
    min_n: int = 50,
    min_r2: float = 0.9,
) -> FADInfResults:
    """
    FAD-inf computed from precomputed embeddings. Mirrors `FrechetAudioDistance.score_inf`
    but skips all audio loading and per-file caching.

    Accepts numpy arrays, torch tensors, or `AudioTensor` instances. Embeddings must be
    shaped `(n_samples, n_features)` (e.g. the output of `get_embedding_directory` /
    `embed_directory`).

    `min_n` anchors the *low* end of the subset-size sweep used for the `1/n`
    extrapolation. It must stay fixed (not scale with the target size) so that scores
    computed from differently-sized target sets remain comparable; a longer lever arm on
    the `1/n` axis also yields a more reliable intercept. `min_r2` is the threshold below
    which the linear fit is considered unreliable (warns rather than raising).
    """

    ref = _as_embedding_array(reference_embeddings)
    tgt = _as_embedding_array(target_embeddings)

    mu_base, cov_base = calc_embd_statistics(ref)

    max_n = tgt.shape[0]
    # Clamp the low anchor to the available samples, but keep it strictly below max_n so
    # the sweep spans a real range of `1/n` (a degenerate single-point sweep makes the
    # polyfit meaningless and silently returns the biased FAD@max_n).
    lo = min(min_n, max_n)
    ns = sorted(set(int(n) for n in np.linspace(lo, max_n, steps)))
    if len(ns) < 2:
        raise ValueError(
            f"FAD-inf needs at least 2 distinct subset sizes for the 1/n fit, got {ns} "
            f"(target has {max_n} samples, min_n={min_n}). Provide a larger target set."
        )

    results: list[list[float]] = []
    with Progress(*Progress.get_default_columns(), TextColumn("current={task.fields[current]}")) as progress:
        progress_task = progress.add_task("FAD", total=len(ns), current=None)
        for n in ns:
            progress.update(progress_task, advance=1.0, current=n)
            # `replace=True` only matters below max_n; at n == max_n we use the full set.
            indices = np.random.choice(max_n, size=n, replace=n < max_n)
            embds_eval = tgt[indices]
            mu_eval, cov_eval = calc_embd_statistics(embds_eval)
            fad_score = calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)
            results.append([n, fad_score])

    ys = np.array(results)
    xs = 1 / ys[:, 0]
    slope, intercept = np.polyfit(xs, ys[:, 1], 1)
    r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum(
        (ys[:, 1] - np.mean(ys[:, 1])) ** 2
    )

    if r2 < min_r2:
        print(
            f"WARNING: FAD-inf linear fit is poor (r2={r2:.3f} < {min_r2}); the "
            f"intercept (score={intercept:.4f}) may be unreliable. Consider a larger "
            f"target set or more steps."
        )

    return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)


def fad_for_serialized_embeddings(reference_path: str, target_path: str) -> FADInfResults:
    reference = load_audio_tensor(reference_path).data
    target = load_audio_tensor(target_path).data
    return fad_from_embeddings(reference, target)
