"""
Train/test splitting utilities for time series data.

Splits respect temporal order and optionally stratify by class label.
"""

from __future__ import annotations

import numpy as np

from ..core.tensor import TimeTensor


def temporal_split(
    ts: TimeTensor,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
) -> tuple[TimeTensor, ...]:
    """Split a TimeTensor preserving temporal order.

    For ragged data, the last ``test_ratio`` fraction of *samples* goes
    to the test set (assumes samples are already ordered).

    Parameters
    ----------
    ts : TimeTensor
    test_ratio : float
        Fraction of samples for the test set.
    val_ratio : float
        Fraction of samples for the validation set (taken before test).

    Returns
    -------
    tuple of TimeTensor
        ``(train,)`` or ``(train, test)`` or ``(train, val, test)``
    """
    n = ts.n_samples
    n_test = max(1, int(n * test_ratio))
    n_val = max(0, int(n * val_ratio))
    n_train = n - n_test - n_val

    if n_train <= 0:
        raise ValueError(f"Not enough samples ({n}) for the requested split ratios.")

    indices = np.arange(n)
    train_idx = indices[:n_train]
    results = [ts[train_idx]]

    if n_val > 0:
        val_idx = indices[n_train : n_train + n_val]
        results.append(ts[val_idx])

    test_idx = indices[n_train + n_val :]
    results.append(ts[test_idx])

    return tuple(results)


def stratified_temporal_split(
    ts: TimeTensor,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
) -> tuple[TimeTensor, ...]:
    """Split a TimeTensor preserving both temporal order and class balance.

    Within each class, the last ``test_ratio`` fraction goes to test.

    Returns
    -------
    tuple of TimeTensor
        ``(train, test)`` or ``(train, val, test)``
    """
    if ts.y is None:
        raise ValueError("stratified_temporal_split requires labels (y).")

    classes = np.unique(ts.y)

    train_indices = []
    val_indices = []
    test_indices = []

    for cls in classes:
        cls_mask = ts.y == cls
        cls_indices = np.where(cls_mask)[0]
        n_cls = len(cls_indices)

        n_test = max(1, int(n_cls * test_ratio))
        n_val = max(0, int(n_cls * val_ratio))
        n_train = n_cls - n_test - n_val

        if n_train <= 0:
            raise ValueError(f"Not enough samples for class {cls} ({n_cls} samples) with the requested split ratios.")

        train_indices.extend(cls_indices[:n_train])
        if n_val > 0:
            val_indices.extend(cls_indices[n_train : n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val :])

    # Sort to maintain overall temporal order within each split
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    results = [ts[np.array(train_indices)]]

    if val_indices:
        results.append(ts[np.array(val_indices)])

    results.append(ts[np.array(test_indices)])

    return tuple(results)


def kfold_temporal(
    ts: TimeTensor,
    n_folds: int = 5,
) -> list[tuple[TimeTensor, TimeTensor]]:
    """Generate temporal K-Fold splits (expanding window).

    Each fold uses all samples up to fold boundary as train, and the
    next block as test.  This prevents future data leakage.

    Returns
    -------
    list of (train, test) TimeTensor pairs
    """
    n = ts.n_samples
    fold_size = n // (n_folds + 1)

    if fold_size < 1:
        raise ValueError(f"Not enough samples ({n}) for {n_folds} folds.")

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, n)

        train = ts[np.arange(train_end)]
        test = ts[np.arange(train_end, test_end)]
        folds.append((train, test))

    return folds
