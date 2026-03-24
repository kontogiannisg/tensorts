"""
Preprocessing transforms: scaling, imputation, normalization.

All transforms operate on the full tensor via vectorized NumPy.
Stateful transforms (scalers) learn parameters on ``fit()`` and apply on ``transform()``.
"""

from __future__ import annotations

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
            self.mean_ = np.mean(all_data, axis=0, keepdims=True)  # (1, C)
            self.std_ = np.std(all_data, axis=0, keepdims=True)  # (1, C)
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
            new_samples = [(s - self.mean_) / (self.std_ + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)
        new_data = (X.data - self.mean_) / (self.std_ + self.eps)
        return X.with_data(new_data)

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [s * (self.std_ + self.eps) + self.mean_ for s in X._samples]
            return X.with_ragged(new_samples)
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

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        lo, hi = self.feature_range
        scale = (hi - lo) / (self.max_ - self.min_ + self.eps)
        if X.is_ragged:
            new_samples = [(s - lo) / scale + self.min_ for s in X._samples]
            return X.with_ragged(new_samples)
        new_data = (X.data - lo) / scale + self.min_
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
            new_samples = [(s - self.median_) / (self.iqr_ + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)
        new_data = (X.data - self.median_) / (self.iqr_ + self.eps)
        return X.with_data(new_data)

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [s * (self.iqr_ + self.eps) + self.median_ for s in X._samples]
            return X.with_ragged(new_samples)
        new_data = X.data * (self.iqr_ + self.eps) + self.median_
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
                result[nans, c] = np.interp(np.where(nans)[0], valid_idx, valid_vals)
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
        encoded = np.array([self._class_to_idx[label] for label in labels])
        return X.with_y(encoded)


# ======================================================================
# MaxAbsScaler
# ======================================================================


class MaxAbsScaler(Transform):
    """Scale each channel by its maximum absolute value.

    Output values lie in [-1, 1].  Preserves sparsity (zeros stay zero).
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.max_abs_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "MaxAbsScaler":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)
            self.max_abs_ = np.max(np.abs(all_data), axis=0, keepdims=True)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                reduce_axes = tuple(i for i in range(X.data.ndim) if i != ch_ax)
            else:
                reduce_axes = None
            self.max_abs_ = np.max(np.abs(X.data), axis=reduce_axes, keepdims=True)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [s / (self.max_abs_ + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(X.data / (self.max_abs_ + self.eps))

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [s * (self.max_abs_ + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(X.data * (self.max_abs_ + self.eps))


# ======================================================================
# Normalizer
# ======================================================================


class Normalizer(Transform):
    """Per-sample normalization (L1, L2, or max norm).

    Stateless — no fitting required.  Each ``(T, C)`` sample is divided
    by its norm so that the norm of the flattened sample equals 1.

    Parameters
    ----------
    norm : str
        ``'l2'`` (default), ``'l1'``, or ``'max'``.
    eps : float
        Small constant to avoid division by zero.
    """

    scope = Scope.PER_SAMPLE
    supports_ragged = True

    def __init__(self, norm: str = "l2", eps: float = 1e-8) -> None:
        super().__init__()
        self.norm = norm
        self.eps = eps

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Normalizer":
        self._is_fitted = True
        return self

    def _compute_norm(self, arr: np.ndarray) -> float:
        if self.norm == "l2":
            return float(np.sqrt(np.sum(arr**2)))
        elif self.norm == "l1":
            return float(np.sum(np.abs(arr)))
        elif self.norm == "max":
            return float(np.max(np.abs(arr)))
        raise ValueError(f"Unknown norm: {self.norm!r}. Use 'l1', 'l2', or 'max'.")

    def transform(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            new_samples = [s / (self._compute_norm(s) + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)

        data = X.data
        b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
        B = data.shape[b_ax]
        # Move batch to front
        perm = [b_ax] + [i for i in range(data.ndim) if i != b_ax]
        transposed = np.transpose(data, perm)
        flat = transposed.reshape(B, -1)

        if self.norm == "l2":
            norms = np.sqrt(np.sum(flat**2, axis=1, keepdims=True))
        elif self.norm == "l1":
            norms = np.sum(np.abs(flat), axis=1, keepdims=True)
        elif self.norm == "max":
            norms = np.max(np.abs(flat), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm!r}. Use 'l1', 'l2', or 'max'.")

        # Reshape norms to broadcast over non-batch dims
        norm_shape = [1] * data.ndim
        norm_shape[b_ax] = B
        norms_broadcast = norms.reshape(norm_shape)
        new_data = data / (norms_broadcast + self.eps)
        return X.with_data(new_data)


# ======================================================================
# PowerTransformer (Yeo-Johnson)
# ======================================================================


class PowerTransformer(Transform):
    """Yeo-Johnson power transform to make data more Gaussian.

    Finds an optimal lambda per channel that makes the marginal
    distribution closest to normal.  Optionally standardises afterward.

    Parameters
    ----------
    standardize : bool
        If True (default), apply zero-mean unit-variance scaling after
        the power transform.
    eps : float
        Small constant for numerical stability.
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(self, standardize: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.standardize = standardize
        self.eps = eps
        self.lambdas_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # -- Yeo-Johnson forward / inverse ------------------------------------

    @staticmethod
    def _yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        if np.abs(lam) > 1e-10:
            out[pos] = ((x[pos] + 1) ** lam - 1) / lam
        else:
            out[pos] = np.log1p(x[pos])
        if np.abs(lam - 2) > 1e-10:
            out[neg] = -((-x[neg] + 1) ** (2 - lam) - 1) / (2 - lam)
        else:
            out[neg] = -np.log1p(-x[neg])
        return out

    @staticmethod
    def _yeo_johnson_inverse(x: np.ndarray, lam: float) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        if np.abs(lam) > 1e-10:
            out[pos] = (x[pos] * lam + 1) ** (1 / lam) - 1
        else:
            out[pos] = np.expm1(x[pos])
        if np.abs(lam - 2) > 1e-10:
            out[neg] = 1 - (-(2 - lam) * x[neg] + 1) ** (1 / (2 - lam))
        else:
            out[neg] = -np.expm1(-x[neg])
        return out

    # -- Lambda optimisation via negative log-likelihood -------------------

    @staticmethod
    def _neg_log_likelihood(lam: float, x: np.ndarray) -> float:
        """Negative log-likelihood of the Yeo-Johnson transform at *lam*."""
        x_t = PowerTransformer._yeo_johnson(x, lam)
        n = len(x_t)
        var = np.var(x_t)
        if var < 1e-16:
            return 1e18
        ll = -n / 2 * np.log(var)
        ll += (lam - 1) * np.sum(np.sign(x) * np.log1p(np.abs(x)))
        return -ll

    def _find_lambda(self, x: np.ndarray) -> float:
        """Find optimal lambda via grid search + refinement."""
        best_lam, best_nll = 0.0, float("inf")
        # Coarse grid
        for lam in np.linspace(-2, 2, 41):
            nll = self._neg_log_likelihood(lam, x)
            if nll < best_nll:
                best_nll = nll
                best_lam = lam
        # Fine grid around best
        for lam in np.linspace(best_lam - 0.1, best_lam + 0.1, 21):
            nll = self._neg_log_likelihood(lam, x)
            if nll < best_nll:
                best_nll = nll
                best_lam = lam
        return float(best_lam)

    # -- fit / transform / inverse -----------------------------------------

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "PowerTransformer":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)  # (T_total, C)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                shape = X.data.shape
                n_ch = shape[ch_ax]
                perm = list(range(X.data.ndim))
                perm.remove(ch_ax)
                perm.append(ch_ax)
                transposed = np.transpose(X.data, perm)
                all_data = transposed.reshape(-1, n_ch)
            else:
                all_data = X.data.ravel()[:, np.newaxis]

        C = all_data.shape[1]
        self.lambdas_ = np.array([self._find_lambda(all_data[:, c]) for c in range(C)])

        if self.standardize:
            transformed = np.column_stack([self._yeo_johnson(all_data[:, c], self.lambdas_[c]) for c in range(C)])
            self.mean_ = np.mean(transformed, axis=0, keepdims=True)
            self.std_ = np.std(transformed, axis=0, keepdims=True)

        self._is_fitted = True
        return self

    def _transform_array(self, arr: np.ndarray) -> np.ndarray:
        C = arr.shape[-1]
        flat = arr.reshape(-1, C).astype(np.float64)
        out = np.column_stack([self._yeo_johnson(flat[:, c], self.lambdas_[c]) for c in range(C)])
        if self.standardize:
            out = (out - self.mean_) / (self.std_ + self.eps)
        return out.reshape(arr.shape)

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._transform_array(s) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(self._transform_array(X.data))

    def _inverse_array(self, arr: np.ndarray) -> np.ndarray:
        C = arr.shape[-1]
        flat = arr.reshape(-1, C).astype(np.float64)
        if self.standardize:
            flat = flat * (self.std_ + self.eps) + self.mean_
        out = np.column_stack([self._yeo_johnson_inverse(flat[:, c], self.lambdas_[c]) for c in range(C)])
        return out.reshape(arr.shape)

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._inverse_array(s) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(self._inverse_array(X.data))


# ======================================================================
# QuantileTransformer
# ======================================================================


class QuantileTransformer(Transform):
    """Map each channel's empirical distribution to uniform or normal.

    Stores per-channel quantile reference values during fit.  Transform
    maps new data into quantile space via interpolation.  Highly robust
    to outliers.

    Parameters
    ----------
    n_quantiles : int
        Number of quantile reference points (capped at n_samples).
    output_distribution : str
        ``'uniform'`` (default) maps to [0, 1].
        ``'normal'`` maps to a standard normal (requires scipy).
    eps : float
        Small constant for numerical stability.
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.eps = eps
        self.quantiles_: np.ndarray | None = None
        self.references_: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "QuantileTransformer":
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            if ch_dim is not None:
                ch_ax = X.axis(ch_dim)
                shape = X.data.shape
                n_ch = shape[ch_ax]
                perm = list(range(X.data.ndim))
                perm.remove(ch_ax)
                perm.append(ch_ax)
                transposed = np.transpose(X.data, perm)
                all_data = transposed.reshape(-1, n_ch)
            else:
                all_data = X.data.ravel()[:, np.newaxis]

        n_samples = all_data.shape[0]
        n_q = min(self.n_quantiles, n_samples)
        self.references_ = np.linspace(0, 1, n_q)
        self.quantiles_ = np.percentile(all_data, self.references_ * 100, axis=0)  # (n_q, C)

        self._is_fitted = True
        return self

    def _transform_channel(self, col: np.ndarray, ch: int) -> np.ndarray:
        """Map a 1-D array to quantile space for channel *ch*."""
        ref_vals = self.quantiles_[:, ch]
        # Interpolate: data values → [0, 1] via reference quantiles
        result = np.interp(col, ref_vals, self.references_)
        result = np.clip(result, 0, 1)
        if self.output_distribution == "normal":
            from scipy.stats import norm

            result = np.clip(result, self.eps, 1 - self.eps)
            result = norm.ppf(result)
        return result

    def _inverse_channel(self, col: np.ndarray, ch: int) -> np.ndarray:
        """Map from quantile space back to data space for channel *ch*."""
        ref_vals = self.quantiles_[:, ch]
        if self.output_distribution == "normal":
            from scipy.stats import norm

            col = norm.cdf(col)
        return np.interp(col, self.references_, ref_vals)

    def _apply_per_channel(self, arr: np.ndarray, fn) -> np.ndarray:
        C = arr.shape[-1]
        flat = arr.reshape(-1, C)
        out = np.column_stack([fn(flat[:, c], c) for c in range(C)])
        return out.reshape(arr.shape)

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._apply_per_channel(s, self._transform_channel) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(self._apply_per_channel(X.data, self._transform_channel))

    def inverse_transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._apply_per_channel(s, self._inverse_channel) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(self._apply_per_channel(X.data, self._inverse_channel))
