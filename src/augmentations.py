"""
adapted from https://github.com/acids-ircam/RAVE/blob/master/rave/dataset.py

`random_phase_mangle` creates an all-pass filter centered around a random frequency,
which preserves spectral magnitude but introduces phase distortion, effectively mangling
the phase while preserving the percieved qualities of the audio.

the phase distortion depends on what frequency the filter is at, thus making it randomized.
"""

from random import random

import numpy as np
from scipy.signal import lfilter
from torch import Tensor, tensor

def _random_angle(min_f: int = 20, max_f: int = 8000, sr: int = 24000) -> float:
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return float(rand)


def _pole_to_z_filter(omega: float, amplitude: float = .9) -> tuple[list[float], list[float]]:
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x: Tensor, min_f: int, max_f: int, amp: float, sr: int) -> Tensor:
    angle = _random_angle(min_f, max_f, sr)
    b, a = _pole_to_z_filter(angle, amp)
    return tensor(lfilter(b, a, x))
