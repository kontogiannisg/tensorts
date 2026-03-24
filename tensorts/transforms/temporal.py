"""
Temporal transforms: windowing, padding, truncation, resampling.

SlidingWindow is the critical ragged → dense boundary — it converts
variable-length sequences into uniform-length windows and handles label
propagation.
"""

from __future__ import annotations

import numpy as np

from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import Scope, Transform

# ======================================================================
# SlidingWindow
# ======================================================================


class SlidingWindow(Transform):
    """Extract fixed-size windows from the time axis.

    Handles both dense ``(B, T, C)`` and ragged inputs.

    Default (``flatten_batch=False``):
        Both dense and ragged → ``(B, K, W, C)`` with dims
        ``(BATCH, WINDOW, TIME, CHANNEL)``.  Ragged requires all samples
        to produce the same number of windows K.

    ``flatten_batch=True``:
        Both dense and ragged → ``(B_total, W, C)`` with dims
        ``(BATCH, TIME, CHANNEL)``.  Batch and window axes are merged.
        Labels are expanded via ``label_strategy`` and ``_window_counts``
        metadata is stored for downstream aggregation.

    Parameters
    ----------
    size : int
        Window length in samples.
    stride : int or None
        Step between windows.  If None, defaults to ``size`` (no overlap).
    label_strategy : str
        How to propagate labels when ``flatten_batch=True``:
        - ``'repeat'``: each window inherits the source sample's label.
        - ``'last'``: only the last window of each sample gets the label (others get -1).
    flatten_batch : bool
        If True, merge batch and window axes into one so the output is
        ``(B_total, W, C)``.  If False (default), output is ``(B, K, W, C)``.
    drop_last : bool
        If True, discard the last window if it's shorter than ``size``.
    """

    supports_ragged = True
    supports_partial_fit = True
    scope = Scope.PER_SAMPLE

    def __init__(
        self,
        size: int,
        stride: int | None = None,
        label_strategy: str = "repeat",
        flatten_batch: bool = False,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.stride = stride or size
        self.label_strategy = label_strategy
        self.flatten_batch = flatten_batch
        self.drop_last = drop_last

    def _num_windows(self, T: int) -> int:
        if self.drop_last:
            return max(0, (T - self.size) // self.stride + 1)
        return max(0, (T - 1) // self.stride + 1)

    def _window_1d(self, arr: np.ndarray) -> np.ndarray:
        """Extract windows from a (T, C) array → (K, W, C)."""
        T = arr.shape[0]
        K = self._num_windows(T)
        if K == 0:
            return np.empty((0, self.size, arr.shape[-1]), dtype=arr.dtype)

        # Use stride tricks for zero-copy windowing
        C = arr.shape[-1]
        strides_t, strides_c = arr.strides
        windows = np.lib.stride_tricks.as_strided(
            arr,
            shape=(K, self.size, C),
            strides=(strides_t * self.stride, strides_t, strides_c),
        )
        return windows.copy()  # copy to make memory contiguous

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "SlidingWindow":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            return self._transform_ragged(X)
        return self._transform_dense(X)

    def _transform_ragged(self, X: TimeTensor) -> TimeTensor:
        all_windows = []
        all_labels = []
        window_counts = []

        for i, sample in enumerate(X._samples):
            wins = self._window_1d(sample)  # (K_i, W, C)
            K_i = wins.shape[0]
            window_counts.append(K_i)
            all_windows.append(wins)

            if X.y is not None:
                label = X.y[i]
                if self.label_strategy == "repeat":
                    all_labels.extend([label] * K_i)
                elif self.label_strategy == "last":
                    all_labels.extend([-1] * (K_i - 1) + [label] if K_i > 0 else [])

        if not all_windows or all(w.shape[0] == 0 for w in all_windows):
            raise ValueError(f"No windows produced. All samples shorter than window size {self.size}.")

        if self.flatten_batch:
            stacked = np.concatenate(all_windows, axis=0)  # (B_total, W, C)
            new_y = np.array(all_labels) if all_labels else None
            meta = {**X.metadata, "_window_counts": window_counts}
            return TimeTensor(
                data=stacked,
                dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                channels=X.channels,
                y=new_y,
                metadata=meta,
            )
        else:
            # All must produce same K
            Ks = [w.shape[0] for w in all_windows]
            if len(set(Ks)) > 1:
                raise ValueError(
                    f"Cannot stack with flatten_batch=False: unequal window counts {Ks}. "
                    "Set flatten_batch=True or pad samples to equal length first."
                )
            stacked = np.stack(all_windows, axis=0)  # (B, K, W, C)
            return TimeTensor(
                data=stacked,
                dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL),
                channels=X.channels,
                y=X.y,
                metadata=X.metadata,
            )

    def _transform_dense(self, X: TimeTensor) -> TimeTensor:
        data = X.data
        t_ax = X.axis(Dim.TIME)

        # Handle arbitrary leading dims by flattening
        # Move TIME to axis 1, keep CHANNEL last
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        # Build permutation: batch dims, TIME, CHANNEL
        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)  # (..., T, C)

        leading_shape = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        flat = transposed.reshape(-1, T, C)  # (N, T, C)

        K = self._num_windows(T)
        windows = np.stack([self._window_1d(flat[i]) for i in range(flat.shape[0])], axis=0)
        # windows: (N, K, W, C)

        new_shape = (*leading_shape, K, self.size, C)
        result = windows.reshape(new_shape)

        # Build new dims: insert WINDOW before TIME
        new_dims = list(X.dims)
        t_dim_idx = new_dims.index(Dim.TIME)
        new_dims.insert(t_dim_idx, Dim.WINDOW)
        # TIME remains but now represents within-window time

        if self.flatten_batch:
            B = result.shape[0]
            K = result.shape[1]
            flat_result = result.reshape(B * K, self.size, C)
            window_counts = [K] * B

            new_y = None
            if X.y is not None:
                all_labels = []
                for i in range(B):
                    label = X.y[i]
                    if self.label_strategy == "repeat":
                        all_labels.extend([label] * K)
                    elif self.label_strategy == "last":
                        all_labels.extend([-1] * (K - 1) + [label])
                new_y = np.array(all_labels)

            meta = {**X.metadata, "_window_counts": window_counts}
            return TimeTensor(
                data=flat_result,
                dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                channels=X.channels,
                y=new_y,
                metadata=meta,
            )

        return X.with_data(result, dims=tuple(new_dims))


# ======================================================================
# Pad
# ======================================================================


class Pad(Transform):
    """Pad sequences along the time axis to a target length.

    Parameters
    ----------
    target_length : int or None
        Desired time-axis length.  If None, pads to the longest in the batch.
    mode : str
        ``'constant'``, ``'edge'``, ``'reflect'``, ``'wrap'``.
    value : float
        Fill value for ``mode='constant'``.
    side : str
        ``'right'`` (default), ``'left'``, or ``'both'``.
    """

    supports_ragged = True
    supports_partial_fit = True
    scope = Scope.PER_SAMPLE

    def __init__(
        self,
        target_length: int | None = None,
        mode: str = "constant",
        value: float = 0.0,
        side: str = "right",
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.mode = mode
        self.value = value
        self.side = side
        self._target: int | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Pad":
        if self.target_length is not None:
            self._target = self.target_length
        elif X.is_ragged:
            self._target = max(s.shape[0] for s in X._samples)
        elif X.has_dim(Dim.TIME):
            self._target = X.data.shape[X.axis(Dim.TIME)]
        else:
            self._target = X.data.shape[1]
        self._is_fitted = True
        return self

    def _pad_array(self, arr: np.ndarray, target: int) -> np.ndarray:
        """Pad a (T, C) or (T,) array along axis 0."""
        T = arr.shape[0]
        if T >= target:
            return arr[:target]

        pad_amount = target - T
        ndim = arr.ndim
        if self.side == "right":
            pad_width = [(0, pad_amount)] + [(0, 0)] * (ndim - 1)
        elif self.side == "left":
            pad_width = [(pad_amount, 0)] + [(0, 0)] * (ndim - 1)
        elif self.side == "both":
            left = pad_amount // 2
            right = pad_amount - left
            pad_width = [(left, right)] + [(0, 0)] * (ndim - 1)
        else:
            raise ValueError(f"Unknown side: {self.side}")

        kwargs = {}
        if self.mode == "constant":
            kwargs["constant_values"] = self.value

        return np.pad(arr, pad_width, mode=self.mode, **kwargs)

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        target = self._target

        if X.is_ragged:
            new_samples = [self._pad_array(s, target) for s in X._samples]
            # Convert to dense
            stacked = np.stack(new_samples, axis=0)  # (B, T, C)
            return TimeTensor(
                data=stacked,
                dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                channels=X.channels,
                y=X.y,
                metadata={**X.metadata, "_padded_from": X.lengths},
            )

        # Dense: pad along TIME axis
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        T = X.data.shape[t_ax]
        if T >= target:
            # Truncate instead
            slices = [slice(None)] * X.data.ndim
            slices[t_ax] = slice(0, target)
            return X.with_data(X.data[tuple(slices)])

        pad_amount = target - T
        pad_width = [(0, 0)] * X.data.ndim
        if self.side == "right":
            pad_width[t_ax] = (0, pad_amount)
        elif self.side == "left":
            pad_width[t_ax] = (pad_amount, 0)
        elif self.side == "both":
            left = pad_amount // 2
            pad_width[t_ax] = (left, pad_amount - left)

        kwargs = {}
        if self.mode == "constant":
            kwargs["constant_values"] = self.value

        padded = np.pad(X.data, pad_width, mode=self.mode, **kwargs)
        return X.with_data(padded)


# ======================================================================
# Truncate
# ======================================================================


class Truncate(Transform):
    """Truncate sequences along the time axis to a target length."""

    supports_ragged = True
    supports_partial_fit = True
    scope = Scope.PER_SAMPLE

    def __init__(self, target_length: int, side: str = "right") -> None:
        super().__init__()
        self.target_length = target_length
        self.side = side

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Truncate":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        T = self.target_length
        if X.is_ragged:
            if self.side == "right":
                new_samples = [s[:T] for s in X._samples]
            else:
                new_samples = [s[-T:] if s.shape[0] > T else s for s in X._samples]
            return X.with_ragged(new_samples)

        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        slices = [slice(None)] * X.data.ndim
        if self.side == "right":
            slices[t_ax] = slice(0, T)
        else:
            slices[t_ax] = slice(-T, None)
        return X.with_data(X.data[tuple(slices)])


# ======================================================================
# Resample
# ======================================================================


class Resample(Transform):
    """Resample the time axis to a new number of samples (via linear interpolation).

    Parameters
    ----------
    target_length : int
        New time-axis length.
    """

    supports_ragged = True
    supports_partial_fit = True
    scope = Scope.PER_SAMPLE

    def __init__(self, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Resample":
        self._is_fitted = True
        return self

    def _resample_2d(self, arr: np.ndarray) -> np.ndarray:
        """Resample a (T, C) array to (target_length, C)."""
        T, C = arr.shape
        if T == self.target_length:
            return arr
        old_x = np.linspace(0, 1, T)
        new_x = np.linspace(0, 1, self.target_length)
        result = np.empty((self.target_length, C), dtype=arr.dtype)
        for c in range(C):
            result[:, c] = np.interp(new_x, old_x, arr[:, c])
        return result

    def transform(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            new_samples = [self._resample_2d(s) for s in X._samples]
            # Now all have same length — can convert to dense
            stacked = np.stack(new_samples, axis=0)
            return TimeTensor(
                data=stacked,
                dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                channels=X.channels,
                y=X.y,
                metadata=X.metadata,
            )

        # Dense: resample along TIME
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else X.data.ndim - 1

        batch_axes = [i for i in range(X.data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(X.data, perm)
        leading = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        flat = transposed.reshape(-1, T, C)

        result = np.stack([self._resample_2d(flat[i]) for i in range(flat.shape[0])], axis=0)
        result = result.reshape(*leading, self.target_length, C)

        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result = np.transpose(result, inv_perm)
        return X.with_data(result)
