# Which columns of a segment file hold the accelerometer axes.
"""Kept apart from `datasets.py` so that the analyses which only need to read signals do not
pull in torch and torchvision. Importing those takes long enough on a shared filesystem to
make `--help` feel broken.
"""

from __future__ import annotations

import numpy as np


def select_scg_channels(x: np.ndarray) -> np.ndarray:
    """Pick the three tri-axial SCG columns out of a (T, C) array.

    The two source datasets store different channel sets, and which columns hold the
    accelerometer is inferred from the total count rather than from any flag in the file:

        7 columns   Dataset II family  -> EKG, then SCG in 1-3, then GCG
        12 columns  Dataset I family   -> SCG in 0-2, then EMG status, ECG, GCG

    Getting this wrong reads gyroscope or ECG channels as accelerometer ones, which trains
    without complaint and produces plausible nonsense.
    """
    if x.shape[1] == 7:
        return x[:, 1:4]
    if x.shape[1] == 12:
        return x[:, 0:3]
    raise ValueError(f"unexpected channel count: {x.shape[1]}")
