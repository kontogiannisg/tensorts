"""
Export utilities for converting TimeTensors back to common formats.
"""

from __future__ import annotations

import numpy as np

from ..core.dims import Dim
from ..core.tensor import TimeTensor


def to_numpy(ts: TimeTensor) -> np.ndarray:
    """Extract the raw ndarray from a dense TimeTensor."""
    return ts.numpy()


def to_sklearn(ts: TimeTensor) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract (X, y) suitable for sklearn estimators.

    Flattens all non-batch dims if the tensor isn't already (B, M).
    """
    data = ts.data

    if ts.has_dim(Dim.BATCH):
        b_ax = ts.axis(Dim.BATCH)
        B = data.shape[b_ax]
        perm = [b_ax] + [i for i in range(data.ndim) if i != b_ax]
        X = np.transpose(data, perm).reshape(B, -1)
    else:
        X = data.ravel()[np.newaxis, :]

    return X, ts.y


def to_dataframe(ts: TimeTensor, sample_index: int = 0):
    """Convert a single sample to a pandas DataFrame.

    For dense TimeTensors, extracts sample ``sample_index`` along the
    batch axis.  Columns are channel names.

    Returns a DataFrame with shape (T, C).
    """
    import pandas as pd

    if ts.is_ragged:
        sample = ts._samples[sample_index]
    else:
        b_ax = ts.axis(Dim.BATCH)
        sample = np.take(ts.data, sample_index, axis=b_ax)

    # sample should be (T, C) or similar
    if sample.ndim > 2:
        # Flatten non-time dims
        sample = sample.reshape(sample.shape[0], -1)

    columns = list(ts.channels.names) if ts.channels else [f"ch_{i}" for i in range(sample.shape[-1])]

    if sample.ndim == 1:
        sample = sample[:, np.newaxis]

    return pd.DataFrame(sample, columns=columns[: sample.shape[-1]])
