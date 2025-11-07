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
from torch import Tensor

def _random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def _pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x: Tensor, min_f, max_f, amp, sr) -> Tensor:
    angle = _random_angle(min_f, max_f, sr)
    b, a = _pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)
