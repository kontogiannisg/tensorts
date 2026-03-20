"""
Feature extraction utilities.

make_statistical_extractor — convenience factory that returns a multi-fn
FeatureExtractor pre-loaded with common statistical functions.
"""

from __future__ import annotations

import numpy as np


def _skew_fn(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std() + 1e-8
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis_fn(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std() + 1e-8
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


_STAT_FN_MAP: dict = {
    "mean":     np.mean,
    "std":      np.std,
    "var":      np.var,
    "min":      np.min,
    "max":      np.max,
    "median":   np.median,
    "rms":      lambda x: float(np.sqrt(np.mean(x ** 2))),
    "range":    np.ptp,
    "energy":   lambda x: float(np.sum(x ** 2)),
    "skew":     _skew_fn,
    "kurtosis": _kurtosis_fn,
    "iqr":      lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
    "zcr":      lambda x: float(np.mean(np.diff(np.signbit(x)).astype(np.float64))),
}

_VALID_STAT_FEATURES = frozenset(_STAT_FN_MAP.keys())


def make_statistical_extractor(
    features: list | None = None,
    input_dim=None,
):
    """Create a :class:`FeatureExtractor` from built-in stats and/or custom functions.

    Each entry in *features* can be:

    - A **string** — name of a built-in statistical function (see below).
    - A **``(name, callable)``** tuple — custom function, ``callable(x: ndarray) → scalar | 1-D array``.
    - A **``(name, callable, kwargs)``** tuple — custom function with keyword arguments.

    Built-in names: ``'mean'``, ``'std'``, ``'var'``, ``'min'``, ``'max'``,
    ``'median'``, ``'rms'``, ``'range'``, ``'energy'``, ``'skew'``,
    ``'kurtosis'``, ``'iqr'``, ``'zcr'``.

    Parameters
    ----------
    features : list, optional
        Mix of strings and ``(name, fn)`` / ``(name, fn, kwargs)`` tuples.
        Default: ``['mean', 'std', 'min', 'max']``.
    input_dim : Dim, optional
        Axis to collapse.  Default: ``Dim.TIME``.

    Returns
    -------
    FeatureExtractor
        Multi-fn mode, consuming ``input_dim``, producing ``Dim.FEATURE``.

    Examples
    --------
    ::

        # Built-ins only
        make_statistical_extractor(["mean", "std", "energy"])

        # Mix built-ins with a custom function
        def fft_rms(x):
            return float(np.sqrt(np.mean(np.abs(np.fft.rfft(x)) ** 2)))

        make_statistical_extractor(["mean", "std", ("fft_rms", fft_rms)])

        # Custom function with kwargs
        def top_k_energy(x, k=5):
            return np.sort(np.abs(np.fft.rfft(x)))[-k:]

        make_statistical_extractor([
            "mean",
            ("top5_energy", top_k_energy, {"k": 5}),
        ])
    """
    from ..core.dims import Dim as _Dim
    from .functional import FeatureExtractor

    if input_dim is None:
        input_dim = _Dim.TIME

    feat_list = features or ["mean", "std", "min", "max"]
    functions = []
    unknown_strings = []

    for entry in feat_list:
        if isinstance(entry, str):
            if entry not in _STAT_FN_MAP:
                unknown_strings.append(entry)
            else:
                functions.append((entry, _STAT_FN_MAP[entry]))
        elif isinstance(entry, tuple) and len(entry) in (2, 3):
            functions.append(entry)
        else:
            raise TypeError(
                f"Each feature must be a str or (name, fn) / (name, fn, kwargs) tuple; "
                f"got {entry!r}."
            )

    if unknown_strings:
        raise ValueError(
            f"Unknown built-in feature(s): {sorted(unknown_strings)}. "
            f"Valid: {sorted(_VALID_STAT_FEATURES)}. "
            "Pass custom functions as (name, callable) tuples."
        )

    return FeatureExtractor(functions=functions, input_dim=input_dim)
