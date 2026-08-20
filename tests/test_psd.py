"""Equivalence of the NumPy Welch implementation with SciPy's.

SciPy is a build-time-only dependency now: it is not imported by the shipped
package, and these tests skip cleanly where it is absent. They exist so the
substitution that removed 112 MB from the bundle stays honest — the jitter
metric's numbers must not move.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.psd import hann, welch

try:
    from scipy import signal as _scipy_signal
except ImportError:                                    # pragma: no cover
    _scipy_signal = None

SKIP = "  SKIP scipy not installed (it is not a runtime dependency)"


def test_hann_matches_scipy():
    if _scipy_signal is None:
        print(SKIP)
        return
    for n in (1, 2, 8, 33, 256):
        ours, theirs = hann(n), _scipy_signal.get_window("hann", n, fftbins=True)
        assert np.allclose(ours, theirs, atol=1e-14), f"hann({n}) diverges"
    print("    hann matches for n in 1, 2, 8, 33, 256")


def test_welch_matches_scipy_across_signals():
    if _scipy_signal is None:
        print(SKIP)
        return
    rng = np.random.default_rng(17)
    cases = {
        "white noise": rng.normal(0, 1, 1024),
        "2 Hz sine": np.sin(2 * np.pi * 2.0 * np.arange(0, 4, 1 / 240)),
        "sine + tremor": (np.sin(2 * np.pi * 2.0 * np.arange(0, 2, 1 / 240))
                          + 0.3 * np.sin(2 * np.pi * 15.0 * np.arange(0, 2, 1 / 240))),
        "ramp": np.linspace(0, 10, 600),
        "constant": np.full(300, 4.2),
    }
    worst = 0.0
    for name, x in cases.items():
        for nperseg in (64, 128, 256):
            f_ours, p_ours = welch(x, fs=240.0, nperseg=min(nperseg, x.size))
            f_sp, p_sp = _scipy_signal.welch(x, fs=240.0, nperseg=min(nperseg, x.size))
            assert np.allclose(f_ours, f_sp, atol=1e-12), f"{name}: frequency bins differ"
            delta = float(np.max(np.abs(p_ours - p_sp)))
            worst = max(worst, delta)
            assert delta < 1e-12, f"{name} @ nperseg={nperseg}: PSD differs by {delta:.2e}"
    print(f"    5 signals x 3 segment lengths, worst deviation {worst:.2e}")


def test_welch_handles_odd_nperseg():
    """Odd lengths have no Nyquist bin, so the one-sided fold differs."""
    if _scipy_signal is None:
        print(SKIP)
        return
    x = np.random.default_rng(3).normal(0, 1, 501)
    for nperseg in (63, 127, 255):
        _, ours = welch(x, fs=60.0, nperseg=nperseg)
        _, theirs = _scipy_signal.welch(x, fs=60.0, nperseg=nperseg)
        assert np.allclose(ours, theirs, atol=1e-12), f"odd nperseg={nperseg} diverges"
    print("    odd nperseg 63, 127, 255 all match")


def test_jitter_metric_unchanged_by_the_substitution():
    """The number a user would actually see."""
    from analyzer.metrics import jitter_ratio
    fps = 240.0
    t = np.arange(0, 2.0, 1 / fps)
    clean = 8 * np.sin(2 * np.pi * 2.0 * t)
    shaky = clean + 0.6 * np.sin(2 * np.pi * 15.0 * t)
    j_clean, j_shaky = jitter_ratio(np.abs(np.diff(clean, prepend=clean[0])) * fps, fps), \
                       jitter_ratio(np.abs(np.diff(shaky, prepend=shaky[0])) * fps, fps)
    assert j_clean < 0.10 and j_shaky > j_clean * 3
    print(f"    jitter still separates clean {j_clean:.4f} from tremor {j_shaky:.4f}")


def test_rejects_bad_arguments():
    try:
        welch(np.zeros((4, 4)))
        raise AssertionError("accepted a 2-D signal")
    except ValueError:
        pass
    try:
        welch(np.zeros(64), nperseg=32, noverlap=32)
        raise AssertionError("accepted noverlap == nperseg")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
