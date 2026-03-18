"""
Preprocessing transforms: scaling, imputation, normalization.

All transforms operate on the full tensor via vectorized NumPy.
Stateful transforms (scalers) learn parameters on ``fit()`` and apply on ``transform()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import Scope, Transform


# ======================================================================
# StandardScaler
# ======================================================================

class StandardScaler(Transform):
    """Zero-mean, unit-variance scaling.

    Computes mean and std over the batch and time axes independently per
    channel.  Works on any schema that has a CHANNEL (or FEATURE) axis.
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "StandardScaler":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)  # (T_total, C)
            self.mean_ = np.mean(all_data, axis=0, keepdims=True)   # (1, C)
            self.std_ = np.std(all_data, axis=0, keepdims=True)     # (1, C)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                reduce_axes = tuple(i for i in range(X.data.ndim) if i != ch_ax)
                self.mean_ = np.mean(X.data, axis=reduce_axes, keepdims=True)
                self.std_ = np.std(X.data, axis=reduce_axes, keepdims=True)
            else:
                self.mean_ = np.mean(X.data, keepdims=True)
                self.std_ = np.std(X.data, keepdims=True)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [
                (s - self.mean_) / (self.std_ + self.eps) for s in X._samples
            ]
            return X.with_ragged(new_samples)
        new_data = (X.data - self.mean_) / (self.std_ + self.eps)
        return X.with_data(new_data)

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        new_data = X.data * (self.std_ + self.eps) + self.mean_
        return X.with_data(new_data)


# ======================================================================
# MinMaxScaler
# ======================================================================

class MinMaxScaler(Transform):
    """Scale to [feature_range[0], feature_range[1]] per channel."""

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8) -> None:
        super().__init__()
        self.feature_range = feature_range
        self.eps = eps
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "MinMaxScaler":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)
            self.min_ = np.min(all_data, axis=0, keepdims=True)
            self.max_ = np.max(all_data, axis=0, keepdims=True)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                reduce_axes = tuple(i for i in range(X.data.ndim) if i != ch_ax)
            else:
                reduce_axes = None
            self.min_ = np.min(X.data, axis=reduce_axes, keepdims=True)
            self.max_ = np.max(X.data, axis=reduce_axes, keepdims=True)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        lo, hi = self.feature_range
        scale = (hi - lo) / (self.max_ - self.min_ + self.eps)

        if X.is_ragged:
            new_samples = [(s - self.min_) * scale + lo for s in X._samples]
            return X.with_ragged(new_samples)
        new_data = (X.data - self.min_) * scale + lo
        return X.with_data(new_data)


# ======================================================================
# RobustScaler
# ======================================================================

class RobustScaler(Transform):
    """Scale using median and IQR (robust to outliers)."""

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0), eps: float = 1e-8) -> None:
        super().__init__()
        self.quantile_range = quantile_range
        self.eps = eps
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "RobustScaler":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)  # (T_total, C)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                # Flatten all non-channel axes
                shape = X.data.shape
                n_ch = shape[ch_ax]
                perm = list(range(X.data.ndim))
                perm.remove(ch_ax)
                perm.append(ch_ax)
                transposed = np.transpose(X.data, perm)
                all_data = transposed.reshape(-1, n_ch)
            else:
                all_data = X.data.ravel()[:, np.newaxis]

        lo, hi = self.quantile_range
        self.median_ = np.median(all_data, axis=0, keepdims=True)
        q_lo = np.percentile(all_data, lo, axis=0, keepdims=True)
        q_hi = np.percentile(all_data, hi, axis=0, keepdims=True)
        self.iqr_ = q_hi - q_lo
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [
                (s - self.median_) / (self.iqr_ + self.eps) for s in X._samples
            ]
            return X.with_ragged(new_samples)
        new_data = (X.data - self.median_) / (self.iqr_ + self.eps)
        return X.with_data(new_data)


# ======================================================================
# Interpolate (imputation)
# ======================================================================

class Interpolate(Transform):
    """Fill NaN values via interpolation along the time axis.

    Parameters
    ----------
    method : str
        ``'linear'``, ``'nearest'``, ``'zero'`` (forward fill), or ``'mean'``.
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, method: str = "linear") -> None:
        super().__init__()
        self.method = method
        self.channel_means_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Interpolate":
        # Store channel means for fallback when entire series is NaN
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)
            self.channel_means_ = np.nanmean(all_data, axis=0)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                reduce_axes = tuple(i for i in range(X.data.ndim) if i != ch_ax)
                self.channel_means_ = np.nanmean(X.data, axis=reduce_axes)
            else:
                self.channel_means_ = np.nanmean(X.data)
        self._is_fitted = True
        return self

    def _interpolate_2d(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate NaNs in a (T, C) array along axis 0."""
        result = arr.copy()
        T, C = result.shape
        for c in range(C):
            col = result[:, c]
            nans = np.isnan(col)
            if not nans.any():
                continue
            if nans.all():
                result[:, c] = self.channel_means_[c] if self.channel_means_ is not None else 0.0
                continue

            valid = ~nans
            valid_idx = np.where(valid)[0]
            valid_vals = col[valid]

            if self.method == "linear":
                result[nans, c] = np.interp(
                    np.where(nans)[0], valid_idx, valid_vals
                )
            elif self.method == "nearest":
                from scipy.interpolate import interp1d
                f = interp1d(valid_idx, valid_vals, kind="nearest", fill_value="extrapolate")
                result[nans, c] = f(np.where(nans)[0])
            elif self.method == "zero":
                # Forward fill
                for i in range(T):
                    if nans[i] and i > 0:
                        result[i, c] = result[i - 1, c]
                # Back-fill remaining leading NaNs
                first_valid = valid_idx[0]
                result[:first_valid, c] = result[first_valid, c]
            elif self.method == "mean":
                result[nans, c] = self.channel_means_[c] if self.channel_means_ is not None else 0.0
        return result

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._interpolate_2d(s) for s in X._samples]
            return X.with_ragged(new_samples)

        data = X.data
        if not np.isnan(data).any():
            return X  # fast path

        # Flatten to (N, T, C) where N = product of all batch-like dims
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        # General strategy: move T and C to last two axes, flatten leading
        perm = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm.extend([t_ax, ch_ax])
        transposed = np.transpose(data, perm)
        orig_leading = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        flat = transposed.reshape(-1, T, C)

        result = np.empty_like(flat)
        for i in range(flat.shape[0]):
            result[i] = self._interpolate_2d(flat[i])

        result = result.reshape(*orig_leading, T, C)
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result = np.transpose(result, inv_perm)
        return X.with_data(result)


# ======================================================================
# LabelEncoder
# ======================================================================

class LabelEncoder(Transform):
    """Encode string or arbitrary labels to integer class indices.

    Operates on ``TimeTensor.y`` rather than the data array.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classes_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "LabelEncoder":
        labels = y if y is not None else X.y
        if labels is None:
            raise ValueError("No labels to encode.")
        self.classes_ = np.unique(labels)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        labels = X.y
        if labels is None:
            return X
        encoded = np.array([self._class_to_idx[l] for l in labels])
        return X.with_y(encoded)
