"""
FunctionTransform — turn any function into a pipeline-compatible transform.

This is the bridge between tensorts and the rest of the scientific Python
ecosystem: scipy.signal, librosa, tsfresh, or any custom function.

The calling convention is scope-based:

  +---------------+---------------------------+---------------------------+
  | Scope         | Function receives         | Library handles           |
  +---------------+---------------------------+---------------------------+
  | PER_CHANNEL   | (T,) 1D array             | Iterates over B and C     |
  | PER_SAMPLE    | (T, C) 2D array           | Iterates over B           |
  | GLOBAL        | Full tensor (B, T, C, ..) | Nothing — user handles    |
  | ELEMENTWISE   | Full tensor               | Nothing — same as GLOBAL  |
  +---------------+---------------------------+---------------------------+

Stateless example (scipy filter):
    FunctionTransform(
        fn=lambda x: scipy.signal.medfilt(x, kernel_size=5),
        scope=Scope.PER_CHANNEL,
    )

Stateful example (sklearn StandardScaler on each sample):
    StatefulFunctionTransform(
        fit_fn=lambda x: {'mean': x.mean(0), 'std': x.std(0)},
        transform_fn=lambda x, state: (x - state['mean']) / state['std'],
        scope=Scope.GLOBAL,
    )
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim, DimSchema
from ..core.tensor import TimeTensor
from .base import Scope, Transform


class FunctionTransform(Transform):
    """Wrap a stateless function as a pipeline Transform.

    Parameters
    ----------
    fn : callable
        The function to apply. Its expected signature depends on *scope*:

        - ``Scope.PER_CHANNEL``: ``fn(x: ndarray[T]) -> ndarray[T']``
          Called once per (sample, channel) combination with a 1D vector.

        - ``Scope.PER_SAMPLE``: ``fn(x: ndarray[T, C]) -> ndarray[T', C'] | ndarray[K]``
          Called once per sample with the full 2D (time, channel) matrix.

        - ``Scope.GLOBAL``: ``fn(x: ndarray[B, T, C, ...]) -> ndarray``
          Called once with the full tensor. User is responsible for
          returning the correct shape.

        - ``Scope.ELEMENTWISE``: same as GLOBAL (function receives full
          tensor, expected to operate element-wise).

    scope : Scope
        Determines how the library dispatches calls to *fn*.
        Defaults to ``Scope.PER_CHANNEL`` — the most common case for
        wrapping external functions (scipy, librosa, etc.).

    output_dims : tuple of Dim, or None
        The semantic dim schema of the output. If None, assumed to be
        the same as the input (shape-preserving transform).  For
        transforms that change dimensionality (e.g., feature extraction),
        you must specify this.

    output_channels : list of str, or None
        Channel names for the output. If None, input channel names are
        preserved.  For transforms that change the channel count or
        produce named features, specify this.

    fn_args : tuple
        Positional arguments forwarded to *fn* after the array argument.

    fn_kwargs : dict
        Keyword arguments forwarded to *fn*.

    name : str or None
        Human-readable name for repr / debugging.

    Examples
    --------
    Wrap scipy's median filter (per-channel, shape-preserving):

    >>> from scipy.signal import medfilt
    >>> smooth = FunctionTransform(
    ...     fn=medfilt,
    ...     scope=Scope.PER_CHANNEL,
    ...     fn_kwargs={'kernel_size': 5},
    ... )

    Wrap librosa's MFCC (per-channel, shape-changing):

    >>> mfcc = FunctionTransform(
    ...     fn=lambda x, sr: librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13).T,
    ...     scope=Scope.PER_CHANNEL,
    ...     output_dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    ...     fn_kwargs={'sr': 16000},
    ... )

    Wrap a multivariate feature extractor (per-sample):

    >>> feat = FunctionTransform(
    ...     fn=my_extractor,  # (T, C) → (K,)
    ...     scope=Scope.PER_SAMPLE,
    ...     output_dims=(Dim.BATCH, Dim.FEATURE),
    ...     output_channels=['f1', 'f2', 'f3'],
    ... )
    """

    supports_ragged = True  # handled internally for PER_CHANNEL and PER_SAMPLE

    def __init__(
        self,
        fn: Callable,
        scope: Scope = Scope.PER_CHANNEL,
        output_dims: tuple[Dim, ...] | None = None,
        output_channels: list[str] | None = None,
        fn_args: tuple = (),
        fn_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.scope = scope
        self._output_dims = output_dims
        self._output_channels = output_channels
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs or {}
        self._name = name or getattr(fn, "__name__", "anonymous")

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "FunctionTransform":
        """Stateless — always fitted."""
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        if self.scope == Scope.PER_CHANNEL:
            return self._apply_per_channel(X)
        elif self.scope == Scope.PER_SAMPLE:
            return self._apply_per_sample(X)
        elif self.scope in (Scope.GLOBAL, Scope.ELEMENTWISE):
            return self._apply_global(X)
        else:
            raise ValueError(f"Unknown scope: {self.scope}")

    # ------------------------------------------------------------------
    # PER_CHANNEL: fn receives (T,) per sample per channel
    # ------------------------------------------------------------------

    def _call_fn(self, arr: np.ndarray) -> np.ndarray:
        """Call self.fn with positional and keyword arguments."""
        return self.fn(arr, *self.fn_args, **self.fn_kwargs)

    def _apply_per_channel(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            return self._apply_per_channel_ragged(X)

        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        # Move TIME and CHANNEL to last two axes, flatten leading dims
        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)  # (..., T, C)

        leading_shape = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        N = int(np.prod(leading_shape)) if leading_shape else 1
        flat = transposed.reshape(N, T, C)

        # Apply fn to first element to discover output shape
        probe = self._call_fn(flat[0, :, 0])
        probe = np.asarray(probe)
        T_out = probe.shape[0] if probe.ndim >= 1 else 1

        # Allocate output and fill
        out = np.empty((N, T_out, C), dtype=probe.dtype)
        out[0, :, 0] = probe.ravel()[:T_out]

        for n in range(N):
            for c in range(C):
                if n == 0 and c == 0:
                    continue  # already computed
                result = self._call_fn(flat[n, :, c])
                result = np.asarray(result)
                out[n, :, c] = result.ravel()[:T_out]

        # Reshape back to original leading dims
        result_shape = (*leading_shape, T_out, C)
        out = out.reshape(result_shape)

        # Undo the permutation
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = np.transpose(out, inv_perm)

        return self._build_output(X, out)

    def _apply_per_channel_ragged(self, X: TimeTensor) -> TimeTensor:
        new_samples = []
        for sample in X._samples:
            # sample: (T_i, C)
            C = sample.shape[-1]
            columns = []
            for c in range(C):
                result = np.asarray(self._call_fn(sample[:, c]))
                if result.ndim == 0:
                    result = result.reshape(1)
                columns.append(result)
            new_samples.append(np.column_stack(columns))

        # Determine output dims/channels
        out_channels = self._resolve_output_channels(X)
        out_dims = self._output_dims if self._output_dims else X.dims
        schema = DimSchema(out_dims)
        return X.with_ragged(new_samples, dims=out_dims, channels=out_channels)

    # ------------------------------------------------------------------
    # PER_SAMPLE: fn receives (T, C) per sample
    # ------------------------------------------------------------------

    def _apply_per_sample(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            return self._apply_per_sample_ragged(X)

        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        # Isolate batch dim, merge middle dims into TIME, keep CHANNEL last
        b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
        B = data.shape[b_ax]

        # Move BATCH to front, CHANNEL to back, flatten rest into TIME
        other_axes = [i for i in range(data.ndim) if i not in (b_ax, ch_ax)]
        perm = [b_ax] + other_axes + [ch_ax]
        transposed = np.transpose(data, perm)  # (B, ..., C)

        middle_shape = transposed.shape[1:-1]
        T_merged = int(np.prod(middle_shape))
        C = transposed.shape[-1]
        flat = transposed.reshape(B, T_merged, C)  # (B, T_merged, C)

        # Apply fn to first sample to discover output shape
        probe = np.asarray(self._call_fn(flat[0]))
        results = [probe]
        for i in range(1, B):
            results.append(np.asarray(self._call_fn(flat[i])))

        out = np.stack(results, axis=0)  # (B, ...)
        return self._build_output(X, out)

    def _apply_per_sample_ragged(self, X: TimeTensor) -> TimeTensor:
        results = [np.asarray(self._call_fn(sample)) for sample in X._samples]

        # Check if all outputs have same shape (can go dense)
        shapes = set(r.shape for r in results)
        out_channels = self._resolve_output_channels(X)
        out_dims = self._output_dims if self._output_dims else X.dims

        if len(shapes) == 1 and results[0].ndim >= 1:
            # Uniform shapes — return dense
            stacked = np.stack(results, axis=0)  # (B, ...)
            return TimeTensor(
                data=stacked,
                dims=out_dims,
                channels=out_channels,
                y=X.y,
                metadata=X.metadata,
            )
        else:
            # Still ragged
            return X.with_ragged(results, dims=out_dims, channels=out_channels)

    # ------------------------------------------------------------------
    # GLOBAL / ELEMENTWISE: fn receives the full tensor
    # ------------------------------------------------------------------

    def _apply_global(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            raise ValueError(
                f"FunctionTransform with scope={self.scope} does not support "
                "ragged input. Use Scope.PER_CHANNEL or Scope.PER_SAMPLE, "
                "or convert to dense first."
            )
        out = np.asarray(self._call_fn(X.data))
        return self._build_output(X, out)

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _resolve_output_channels(self, X: TimeTensor) -> ChannelIndex | list[str] | None:
        if self._output_channels is not None:
            return ChannelIndex(self._output_channels) if isinstance(
                self._output_channels, list
            ) else self._output_channels
        return X.channels

    def _build_output(self, X: TimeTensor, out: np.ndarray) -> TimeTensor:
        """Construct the output TimeTensor with proper dims and channels."""
        out_channels = self._resolve_output_channels(X)
        out_dims = self._output_dims if self._output_dims is not None else X.dims

        # If output has different number of dims than declared, try to infer
        if len(out_dims) != out.ndim:
            # Common case: PER_SAMPLE returned (K,) per sample → stacked to (B, K)
            if out.ndim == 2 and Dim.BATCH in out_dims and Dim.FEATURE in out_dims:
                pass  # already correct
            else:
                # Fall back to input dims if output shape matches
                if out.ndim == len(X.dims):
                    out_dims = X.dims
                else:
                    # Auto-generate dims: BATCH + as many generic dims as needed
                    out_dims = (Dim.BATCH,) + tuple(
                        Dim.FEATURE if i == out.ndim - 2 else Dim.TIME
                        for i in range(out.ndim - 1)
                    )
                    if out.ndim == 2:
                        out_dims = (Dim.BATCH, Dim.FEATURE)

        # Auto-generate channel names if output channel count changed
        if out_channels is not None:
            # Find which axis is the channel/feature axis
            schema = DimSchema(out_dims)
            ch_dim = None
            for d in (Dim.CHANNEL, Dim.FEATURE):
                if schema.has(d):
                    ch_dim = d
                    break
            if ch_dim is not None:
                ch_size = out.shape[schema.axis(ch_dim)]
                if isinstance(out_channels, ChannelIndex) and len(out_channels) != ch_size:
                    out_channels = ChannelIndex.auto(ch_size, prefix="out")
                elif isinstance(out_channels, list) and len(out_channels) != ch_size:
                    out_channels = ChannelIndex.auto(ch_size, prefix="out")

        return X.with_data(out, dims=out_dims, channels=out_channels)

    def get_params(self) -> dict[str, Any]:
        return {
            "fn": self._name,
            "scope": self.scope.name,
            "output_dims": self._output_dims,
            "output_channels": self._output_channels,
        }

    def __repr__(self) -> str:
        return (
            f"FunctionTransform(fn={self._name}, scope={self.scope.name}"
            f"{', output_dims=' + str(self._output_dims) if self._output_dims else ''})"
        )


class StatefulFunctionTransform(Transform):
    """Wrap a stateful function pair as a pipeline Transform.

    For when the external function needs a fit step (learning parameters
    from training data) before it can transform.

    Parameters
    ----------
    fit_fn : callable
        ``fit_fn(x) -> state``

        Called during ``fit()`` to learn parameters. Receives data shaped
        according to *scope* (same rules as ``FunctionTransform``).
        Must return a state object (dict, tuple, any picklable object).

    transform_fn : callable
        ``transform_fn(x, state) -> ndarray``

        Called during ``transform()`` with the data and the state learned
        by *fit_fn*.

    scope : Scope
        Same dispatch rules as ``FunctionTransform``.

    fit_scope : Scope or None
        If different from *scope*, use this scope for the fit step.
        Common case: fit on the global tensor (to compute statistics),
        transform per-channel.  If None, uses *scope*.

    output_dims, output_channels, fn_args, fn_kwargs, name :
        Same as ``FunctionTransform``.

    Examples
    --------
    Wrap an external library's scaler:

    >>> sfn = StatefulFunctionTransform(
    ...     fit_fn=lambda x: {'mu': x.mean(axis=0), 'sigma': x.std(axis=0)},
    ...     transform_fn=lambda x, s: (x - s['mu']) / (s['sigma'] + 1e-8),
    ...     scope=Scope.GLOBAL,
    ... )

    Per-channel with fitted state (e.g., adaptive filter coefficients):

    >>> sfn = StatefulFunctionTransform(
    ...     fit_fn=lambda x: {'threshold': np.percentile(np.abs(x), 95)},
    ...     transform_fn=lambda x, s: np.clip(x, -s['threshold'], s['threshold']),
    ...     scope=Scope.PER_CHANNEL,
    ... )
    """

    supports_ragged = True

    def __init__(
        self,
        fit_fn: Callable,
        transform_fn: Callable,
        scope: Scope = Scope.PER_CHANNEL,
        fit_scope: Scope | None = None,
        output_dims: tuple[Dim, ...] | None = None,
        output_channels: list[str] | None = None,
        fn_args: tuple = (),
        fn_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.fit_fn = fit_fn
        self.transform_fn = transform_fn
        self.scope = scope
        self.fit_scope = fit_scope or scope
        self._output_dims = output_dims
        self._output_channels = output_channels
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs or {}
        self._name = name or "stateful"
        self._state: Any = None
        self._channel_states: list[Any] | None = None

    def _call_fit(self, arr: np.ndarray) -> Any:
        return self.fit_fn(arr, *self.fn_args, **self.fn_kwargs)

    def _call_transform(self, arr: np.ndarray, state: Any) -> np.ndarray:
        return self.transform_fn(arr, state, *self.fn_args, **self.fn_kwargs)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "StatefulFunctionTransform":
        if self.fit_scope == Scope.PER_CHANNEL:
            self._fit_per_channel(X)
        elif self.fit_scope == Scope.PER_SAMPLE:
            self._fit_per_sample(X)
        elif self.fit_scope in (Scope.GLOBAL, Scope.ELEMENTWISE):
            self._fit_global(X)
        self._is_fitted = True
        return self

    def _fit_global(self, X: TimeTensor) -> None:
        if X.is_ragged:
            data = np.concatenate(X._samples, axis=0)
        else:
            data = X.data
        self._state = self._call_fit(data)

    def _fit_per_channel(self, X: TimeTensor) -> None:
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)  # (T_total, C)
        else:
            # Flatten to (T_total, C)
            ch_dim = X._find_channel_dim(X.schema)
            ch_ax = X.axis(ch_dim) if ch_dim is not None else X.data.ndim - 1
            t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1

            perm = [i for i in range(X.data.ndim) if i not in (t_ax, ch_ax)]
            perm += [t_ax, ch_ax]
            transposed = np.transpose(X.data, perm)
            C = transposed.shape[-1]
            all_data = transposed.reshape(-1, C)

        C = all_data.shape[-1]
        self._channel_states = [self._call_fit(all_data[:, c]) for c in range(C)]

    def _fit_per_sample(self, X: TimeTensor) -> None:
        # Fit on the concatenation of all samples (as if one big sample)
        if X.is_ragged:
            data = np.concatenate(X._samples, axis=0)
        else:
            b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
            # Merge batch into time
            perm = [b_ax] + [i for i in range(X.data.ndim) if i != b_ax]
            transposed = np.transpose(X.data, perm)
            B = transposed.shape[0]
            data = transposed.reshape(-1, *transposed.shape[2:])
        self._state = self._call_fit(data)

    # ------------------------------------------------------------------
    # Transform — delegates to a FunctionTransform with state baked in
    # ------------------------------------------------------------------

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()

        if self.scope == Scope.PER_CHANNEL:
            return self._transform_per_channel(X)
        elif self.scope == Scope.PER_SAMPLE:
            return self._transform_per_sample(X)
        elif self.scope in (Scope.GLOBAL, Scope.ELEMENTWISE):
            return self._transform_global(X)
        raise ValueError(f"Unknown scope: {self.scope}")

    def _transform_global(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            raise ValueError("GLOBAL scope does not support ragged input.")
        out = np.asarray(self._call_transform(X.data, self._state))
        out_dims = self._output_dims or X.dims
        out_ch = self._output_channels or X.channels
        return X.with_data(out, dims=out_dims, channels=out_ch)

    def _transform_per_channel(self, X: TimeTensor) -> TimeTensor:
        states = self._channel_states
        if states is None:
            raise RuntimeError("Per-channel states not fitted.")

        if X.is_ragged:
            new_samples = []
            for sample in X._samples:
                C = sample.shape[-1]
                columns = [
                    np.asarray(self._call_transform(sample[:, c], states[c]))
                    for c in range(C)
                ]
                # Handle 0-d or 1-d returns
                columns = [col.ravel() if col.ndim == 0 else col for col in columns]
                new_samples.append(np.column_stack(columns))
            out_dims = self._output_dims or X.dims
            out_ch = self._output_channels or X.channels
            return X.with_ragged(new_samples, dims=out_dims, channels=out_ch)

        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)

        leading_shape = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        N = int(np.prod(leading_shape)) if leading_shape else 1
        flat = transposed.reshape(N, T, C)

        probe = np.asarray(self._call_transform(flat[0, :, 0], states[0]))
        T_out = probe.shape[0] if probe.ndim >= 1 else 1

        out = np.empty((N, T_out, C), dtype=probe.dtype)
        out[0, :, 0] = probe.ravel()[:T_out]

        for n in range(N):
            for c in range(C):
                if n == 0 and c == 0:
                    continue
                result = np.asarray(self._call_transform(flat[n, :, c], states[c]))
                out[n, :, c] = result.ravel()[:T_out]

        result_shape = (*leading_shape, T_out, C)
        out = out.reshape(result_shape)

        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = np.transpose(out, inv_perm)

        out_dims = self._output_dims or X.dims
        out_ch = self._output_channels or X.channels
        return X.with_data(out, dims=out_dims, channels=out_ch)

    def _transform_per_sample(self, X: TimeTensor) -> TimeTensor:
        state = self._state

        if X.is_ragged:
            results = [np.asarray(self._call_transform(s, state)) for s in X._samples]
            shapes = set(r.shape for r in results)
            out_dims = self._output_dims or X.dims
            out_ch = self._output_channels or X.channels
            if len(shapes) == 1:
                return TimeTensor(
                    data=np.stack(results, axis=0),
                    dims=out_dims,
                    channels=out_ch,
                    y=X.y,
                    metadata=X.metadata,
                )
            return X.with_ragged(results, dims=out_dims, channels=out_ch)

        data = X.data
        b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
        B = data.shape[b_ax]

        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
        other_axes = [i for i in range(data.ndim) if i not in (b_ax, ch_ax)]
        perm = [b_ax] + other_axes + [ch_ax]
        transposed = np.transpose(data, perm)

        middle_shape = transposed.shape[1:-1]
        T_merged = int(np.prod(middle_shape))
        C = transposed.shape[-1]
        flat = transposed.reshape(B, T_merged, C)

        results = [np.asarray(self._call_transform(flat[i], state)) for i in range(B)]
        out = np.stack(results, axis=0)

        out_dims = self._output_dims or X.dims
        out_ch = self._output_channels or X.channels
        return X.with_data(out, dims=out_dims, channels=out_ch)

    def get_params(self) -> dict[str, Any]:
        return {"name": self._name, "scope": self.scope.name}

    def __repr__(self) -> str:
        return (
            f"StatefulFunctionTransform({self._name}, scope={self.scope.name})"
        )
