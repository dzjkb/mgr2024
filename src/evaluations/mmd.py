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

def mmd(z_tilde, z, kernel='imq'):
    def gaussian(d, var=16.):
        return torch.exp(- d / var).sum(1).sum(0)

    def inverse_multiquadratics(d, var=16.):
        """
        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return (var / (var + d)).sum(1).sum(0)
    if kernel == 'imq':

        k = inverse_multiquadratics
    elif kernel == 'gaussian':
        k = gaussian
    else:
        raise AttributeError(f'Kernel type {kernel} not understood. Available: [gaussian, imq]')

    batch_size = z_tilde.size(0)
    zz_ker = _compute_kernel(z, z, k)
    z_tilde_z_tilde_ker = _compute_kernel(z_tilde, z_tilde, k)
    z_z_tilde_ker = _compute_kernel(z, z_tilde, k)

    first_coefs = 1. / (batch_size * (batch_size - 1))
    second_coef = 2 / (batch_size * batch_size)
    mmd_distance = (first_coefs * zz_ker
           + first_coefs * z_tilde_z_tilde_ker
           - second_coef * z_z_tilde_ker)
    return mmd_distance