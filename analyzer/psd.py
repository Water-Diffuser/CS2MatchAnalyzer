"""Welch's method in NumPy.

The only thing this project used SciPy for was `signal.welch`, and SciPy adds
roughly 112 MB to a frozen bundle — more than the rest of the application and
its OpenCV dependency combined. For a desktop tool people download, that trade
is not worth one function.

Verified against `scipy.signal.welch` to within 1e-12 in tests/test_psd.py, so
the jitter metric's numbers are unchanged by the substitution.
"""
from __future__ import annotations

import numpy as np


def hann(n: int, *, sym: bool = False) -> np.ndarray:
    """Hann window.

    Defaults to the periodic (`sym=False`) form, which is what spectral
    estimation wants — the symmetric form is for filter design, and using it
    here would bias the estimate slightly at every segment boundary.
    """
    if n < 1:
        return np.ones(0)
    if n == 1:
        return np.ones(1)
    divisor = n if not sym else n - 1
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / divisor)


def welch(x: np.ndarray, fs: float = 1.0, nperseg: int | None = None,
          noverlap: int | None = None, detrend: str | None = "constant"
          ) -> tuple[np.ndarray, np.ndarray]:
    """Estimate power spectral density by averaging windowed periodograms.

    Matches `scipy.signal.welch` for the one-sided, density-scaled, Hann-window
    case this project needs. Returns (frequencies, PSD).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("welch expects a 1-D signal")

    n = x.size
    if nperseg is None:
        nperseg = min(n, 256)
    nperseg = int(min(nperseg, n))
    if nperseg < 1:
        raise ValueError("nperseg must be at least 1")
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - int(noverlap)
    if step < 1:
        raise ValueError("noverlap must be less than nperseg")

    win = hann(nperseg)
    # Density scaling: normalizing by the window's power keeps the result
    # independent of window choice, so PSD values stay comparable.
    scale = 1.0 / (fs * np.sum(win ** 2))

    starts = range(0, n - nperseg + 1, step)
    segments = []
    for start in starts:
        seg = x[start:start + nperseg]
        if detrend == "constant":
            seg = seg - seg.mean()
        elif detrend == "linear":
            t = np.arange(seg.size)
            seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
        spectrum = np.fft.rfft(seg * win)
        segments.append((spectrum.real ** 2 + spectrum.imag ** 2) * scale)

    if not segments:
        return np.zeros(0), np.zeros(0)

    psd = np.mean(segments, axis=0)
    # One-sided: fold the negative frequencies in by doubling everything except
    # DC and, when nperseg is even, Nyquist — neither of which has a mirror.
    if psd.size > 1:
        end = -1 if nperseg % 2 == 0 else None
        psd[1:end] *= 2.0

    return np.fft.rfftfreq(nperseg, 1.0 / fs), psd
