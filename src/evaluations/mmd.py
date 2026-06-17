# adapted from https://github.com/SonyCSLParis/DrumGAN/blob/master/evaluation/metrics/maximum_mean_discrepancy.py
# type: ignore

import torch


def _compute_kernel(x, y, k):
    # Pairwise squared Euclidean distances via the algebraic identity
    # ‖x - y‖² = ‖x‖² + ‖y‖² - 2·x·yᵀ. Avoids materializing the (N, M, D)
    # broadcast tensor that the expand+subtract approach allocates.
    assert x.size(1) == y.size(1)
    x_sq = (x * x).sum(dim=1, keepdim=True)         # (N, 1)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()      # (1, M)
    distances = x_sq + y_sq - 2.0 * (x @ y.t())     # (N, M)
    distances = distances.clamp_min(0.0)             # numerical safety
    return k(distances)


def mmd(z_tilde, z, kernel='imq', normalize=True):
    """
    Unbiased estimator of the squared Maximum Mean Discrepancy between two
    embedding sets, with an inverse-multiquadratic (or gaussian) kernel.

    :param z_tilde: (M, D) embeddings of the first set (e.g. generated)
    :param z:       (N, D) embeddings of the second set (e.g. reference)
    :param normalize: if True, L2-normalize embeddings before the kernel so the
        score is invariant to embedding magnitude (CLAP embeddings are not
        guaranteed unit-norm depending on the extraction path).

    The two sets may have different sizes. The within-set terms exclude the
    diagonal (the unbiased estimator drops the k(x_i, x_i) self-terms, each
    equal to kernel(0)=1), and each term is normalized by its own set size:
        1/(N(N-1)) for z, 1/(M(M-1)) for z_tilde, 2/(NM) for the cross term.
    """
    def gaussian(d, var=16.):
        return torch.exp(- d / var)

    def inverse_multiquadratics(d, var=16.):
        """
        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return var / (var + d)

    if kernel == 'imq':
        k = inverse_multiquadratics
    elif kernel == 'gaussian':
        k = gaussian
    else:
        raise AttributeError(f'Kernel type {kernel} not understood. Available: [gaussian, imq]')

    if normalize:
        z = torch.nn.functional.normalize(z, dim=1)
        z_tilde = torch.nn.functional.normalize(z_tilde, dim=1)

    n = z.size(0)
    m = z_tilde.size(0)
    assert n > 1 and m > 1, f"need at least 2 samples per set, got n={n}, m={m}"

    zz_ker = _compute_kernel(z, z, k)                       # (N, N)
    z_tilde_z_tilde_ker = _compute_kernel(z_tilde, z_tilde, k)  # (M, M)
    z_z_tilde_ker = _compute_kernel(z, z_tilde, k)          # (N, M)

    # exclude the diagonal self-terms from the within-set sums
    zz_sum = zz_ker.sum() - zz_ker.diagonal().sum()
    z_tilde_z_tilde_sum = z_tilde_z_tilde_ker.sum() - z_tilde_z_tilde_ker.diagonal().sum()
    cross_sum = z_z_tilde_ker.sum()

    mmd_distance = (zz_sum / (n * (n - 1))
                    + z_tilde_z_tilde_sum / (m * (m - 1))
                    - 2.0 * cross_sum / (n * m))
    return mmd_distance
