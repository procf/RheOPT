"""Wake diagnostics: dominant shedding frequency and RMS of the transverse velocity along the wake.

Coordinates are those of the stored wake mesh (inlet at x = -4, triangle at x = -3); plots shift x by +4.
Predicted frames are 0.2 s apart.
"""
import numpy as np

DT = 0.2                                   # s between predicted frames
WAKE_X_START, WAKE_Y_HALF = -3.0, 0.5      # wake region: x > -3 (1 cm behind the inlet), |y| < 0.5
X_EDGES = np.round(np.arange(-4.0, 0.0 + 1e-9, 0.1), 2)
X_SHIFT = 4.0
REPRESENTATIVE_RE = 94.121                 # case used for the RMS profile (and the paper's field figures)


def wake_mask(pos):
    return (pos[:, 0] > WAKE_X_START) & (np.abs(pos[:, 1]) < WAKE_Y_HALF)


def peak_bin(series):
    """Index k >= 1 of the dominant non-zero frequency of the wake-averaged power spectrum; series: (frames, nodes).
    The frequency is k / (frames * DT)."""
    spec = (np.abs(np.fft.rfft(series - series.mean(0, keepdims=True), axis=0)) ** 2).mean(1)
    return int(np.argmax(spec[1:]) + 1)


def peak_frequency(series, dt=DT):
    return peak_bin(series) / (series.shape[0] * dt)


def rms_profile(vy, x):
    """RMS fluctuation of v_y in 0.1-wide x bins: sqrt of the node-averaged temporal variance. vy: (frames, nodes).
    Returns (bin centres shifted by X_SHIFT, rms)."""
    var = vy.var(axis=0)
    centres, prof = [], []
    for a, b in zip(X_EDGES[:-1], X_EDGES[1:]):
        m = (x >= a) & (x < b)
        if m.any():
            centres.append(0.5 * (a + b) + X_SHIFT)
            prof.append(np.sqrt(var[m].mean()))
    return np.array(centres), np.array(prof)
