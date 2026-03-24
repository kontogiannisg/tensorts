"""
FunctionTransform — wrap signal-processing functions into pipeline transforms.
FeatureExtractor  — wrap one or many feature-extraction functions that change
                    dimensionality, or stack multiple functions à la FeatureUnion.

FunctionTransform is for shape-preserving (or length-changing) operations:
  fn(T,) -> (T',) per channel.  The dim schema stays the same.

FeatureExtractor operates in two modes:

  Single-fn mode  (fn=...)
    One function, arbitrary output shape.  The consumed axis is replaced by
    ``output_dims``.  Suitable for spectrograms, STFT, wavelets, etc.

  Multi-fn mode  (functions=[...])
    A list of ``(name, fn)`` or ``(name, fn, kwargs)`` pairs, each returning
    a scalar or 1-D array per channel.  All results are concatenated along a
    new ``Dim.FEATURE`` axis, replacing ``input_dim``.  Acts like sklearn's
    FeatureUnion and replaces StatisticalFeatures / FrameFeatureBank for
    non-FFT-cached use cases.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import Scope, Transform

# ======================================================================
# FunctionTransform — shape-preserving signal processing wrapper
# ======================================================================


class FunctionTransform(Transform):
    """Wrap a stateless function as a pipeline Transform.

    The function receives a 1D, 2D, or full tensor depending on *scope*
    and must return the same rank (shape may differ in size but not ndim).

    Parameters
    ----------
    fn : callable
        - PER_CHANNEL: fn(x: 1D) -> 1D (same or different length)
        - PER_SAMPLE:  fn(x: 2D) -> 2D (same or different shape)
        - GLOBAL:      fn(x: full tensor) -> same-rank tensor
    scope : Scope
    fn_args : tuple — extra positional args forwarded to fn
    fn_kwargs : dict — extra keyword args forwarded to fn
    name : str or None — for repr

    Note
    ----
    FunctionTransform preserves the dim schema — it does not rename axes.
    If your function changes the semantic meaning of an axis (e.g., FFT
    converts TIME → FREQUENCY), use ``FeatureExtractor`` with appropriate
    ``output_dims`` instead, so downstream transforms see the correct schema.
    """

    supports_ragged = True

    def __init__(
        self,
        fn: Callable,
        scope: Scope = Scope.PER_CHANNEL,
        fn_args: tuple = (),
        fn_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.scope = scope
        self.fn_args = fn_args
        self.fn_kwargs = {} if fn_kwargs is None else fn_kwargs
        self.name = name or getattr(fn, "__name__", "anonymous")

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "FunctionTransform":
        self._is_fitted = True
        return self

    def _call_fn(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(arr, *self.fn_args, **self.fn_kwargs))

    def transform(self, X: TimeTensor) -> TimeTensor:
        if self.scope == Scope.PER_CHANNEL:
            return self._apply_per_channel(X)
        elif self.scope == Scope.PER_SAMPLE:
            return self._apply_per_sample(X)
        elif self.scope in (Scope.GLOBAL, Scope.ELEMENTWISE):
            return self._apply_global(X)
        raise ValueError(f"Unknown scope: {self.scope}")

    # -- PER_CHANNEL: fn receives (T,) -> (T',) ----------------------------

    def _apply_per_channel(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            return self._apply_per_channel_ragged(X)

        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)  # (..., T, C)

        leading_shape = transposed.shape[:-2]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        N = int(np.prod(leading_shape)) if leading_shape else 1
        flat = transposed.reshape(N, T, C)

        probe = self._call_fn(flat[0, :, 0])
        T_out = probe.shape[0] if probe.ndim >= 1 else 1

        out = np.empty((N, T_out, C), dtype=probe.dtype)
        out[0, :, 0] = probe.ravel()[:T_out]
        for n in range(N):
            for c in range(C):
                if n == 0 and c == 0:  # Already computed to extract T dimension
                    continue
                result = self._call_fn(flat[n, :, c])
                out[n, :, c] = np.asarray(result).ravel()[:T_out]

        out = out.reshape(*leading_shape, T_out, C)
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = np.transpose(out, inv_perm)
        return X.with_data(out)

    def _apply_per_channel_ragged(self, X: TimeTensor) -> TimeTensor:
        new_samples = []
        for sample in X._samples:
            C = sample.shape[-1]
            cols = [self._call_fn(sample[:, c]) for c in range(C)]
            cols = [r.ravel() if r.ndim == 0 else r for r in cols]
            new_samples.append(np.column_stack(cols))
        return X.with_ragged(new_samples)

    # -- PER_SAMPLE: fn receives (T, C) -> (T', C') -----------------------

    def _apply_per_sample(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            return self._apply_per_sample_ragged(X)

        data = X.data
        b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
        B = data.shape[b_ax]

        other_axes = [i for i in range(data.ndim) if i not in (b_ax, ch_ax)]
        perm = [b_ax] + other_axes + [ch_ax]
        transposed = np.transpose(data, perm)
        T_merged = int(np.prod(transposed.shape[1:-1]))
        C = transposed.shape[-1]
        flat = transposed.reshape(B, T_merged, C)

        results = [self._call_fn(flat[i]) for i in range(B)]
        out = np.stack(results, axis=0)
        return X.with_data(out)

    def _apply_per_sample_ragged(self, X: TimeTensor) -> TimeTensor:
        results = [self._call_fn(s) for s in X._samples]
        shapes = set(r.shape for r in results)
        if len(shapes) == 1:
            stacked = np.stack(results, axis=0)
            return TimeTensor(
                data=stacked,
                dims=X.dims,
                channels=X.channels,
                y=X.y,
                metadata=X.metadata,
            )
        return X.with_ragged(results)

    # -- GLOBAL: fn receives full tensor -----------------------------------

    def _apply_global(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            raise ValueError("GLOBAL scope does not support ragged input.")
        return X.with_data(self._call_fn(X.data))

    def __repr__(self) -> str:
        return f"FunctionTransform(fn={self.name}, scope={self.scope.name})"


# ======================================================================
# StatefulFunctionTransform
# ======================================================================


class StatefulFunctionTransform(Transform):
    """Wrap a stateful function pair (fit_fn, transform_fn) as a Transform.

    fit_fn(x) -> state     (called during fit)
    transform_fn(x, state) -> ndarray  (called during transform)

    Scope semantics match FunctionTransform.  For PER_CHANNEL, a separate
    state is learned per channel.
    """

    supports_ragged = True

    def __init__(
        self, fit_fn, transform_fn, scope=Scope.PER_CHANNEL, fit_scope=None, fn_args=(), fn_kwargs=None, name=None
    ):
        super().__init__()
        self.fit_fn = fit_fn
        self.transform_fn = transform_fn
        self.scope = scope
        self.fit_scope = fit_scope or scope
        self.fn_args = fn_args
        self.fn_kwargs = {} if fn_kwargs is None else fn_kwargs
        self.name = name or "stateful"
        self._state = None
        self._channel_states = None

    def _call_fit(self, arr):
        return self.fit_fn(arr, *self.fn_args, **self.fn_kwargs)

    def _call_transform(self, arr, state):
        return np.asarray(self.transform_fn(arr, state, *self.fn_args, **self.fn_kwargs))

    def fit(self, X, y=None):
        if self.fit_scope == Scope.PER_CHANNEL:
            self._fit_per_channel(X)
        elif self.fit_scope == Scope.PER_SAMPLE:
            self._fit_per_sample(X)
        else:
            self._fit_global(X)
        self._is_fitted = True
        return self

    def _fit_global(self, X):
        data = np.concatenate(X._samples, axis=0) if X.is_ragged else X.data
        self._state = self._call_fit(data)

    def _fit_per_channel(self, X):
        if X.is_ragged:
            all_data = np.concatenate(X._samples, axis=0)
        else:
            ch_dim = X._find_channel_dim(X.schema)
            ch_ax = X.axis(ch_dim) if ch_dim is not None else X.data.ndim - 1
            t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
            perm = [i for i in range(X.data.ndim) if i not in (t_ax, ch_ax)] + [t_ax, ch_ax]
            transposed = np.transpose(X.data, perm)
            all_data = transposed.reshape(-1, transposed.shape[-1])
        C = all_data.shape[-1]
        self._channel_states = [self._call_fit(all_data[:, c]) for c in range(C)]

    def _fit_per_sample(self, X):
        if X.is_ragged:
            data = np.concatenate(X._samples, axis=0)
        else:
            b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
            perm = [b_ax] + [i for i in range(X.data.ndim) if i != b_ax]
            data = np.transpose(X.data, perm).reshape(-1, *np.transpose(X.data, perm).shape[2:])
        self._state = self._call_fit(data)

    def transform(self, X):
        self._check_fitted()
        if self.scope == Scope.PER_CHANNEL:
            return self._transform_per_channel(X)
        elif self.scope == Scope.PER_SAMPLE:
            return self._transform_per_sample(X)
        return self._transform_global(X)

    def _transform_global(self, X):
        if X.is_ragged:
            raise ValueError("GLOBAL scope does not support ragged input.")
        return X.with_data(self._call_transform(X.data, self._state))

    def _transform_per_channel(self, X):
        states = self._channel_states
        if X.is_ragged:
            new_samples = []
            for sample in X._samples:
                C = sample.shape[-1]
                cols = [self._call_transform(sample[:, c], states[c]) for c in range(C)]
                cols = [r.ravel() if r.ndim == 0 else r for r in cols]
                new_samples.append(np.column_stack(cols))
            return X.with_ragged(new_samples)

        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)
        leading = transposed.shape[:-2]
        T, C = transposed.shape[-2], transposed.shape[-1]
        N = int(np.prod(leading)) if leading else 1
        flat = transposed.reshape(N, T, C)

        probe = self._call_transform(flat[0, :, 0], states[0])
        T_out = probe.shape[0] if probe.ndim >= 1 else 1
        out = np.empty((N, T_out, C), dtype=probe.dtype)
        out[0, :, 0] = probe.ravel()[:T_out]
        for n in range(N):
            for c in range(C):
                if n == 0 and c == 0:
                    continue
                out[n, ..., c] = self._call_transform(flat[n, :, c], states[c]).ravel()[:T_out]
        out = out.reshape(*leading, T_out, C)
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = np.transpose(out, inv_perm)
        return X.with_data(out)

    def _transform_per_sample(self, X):
        state = self._state
        if X.is_ragged:
            results = [self._call_transform(s, state) for s in X._samples]
            shapes = set(r.shape for r in results)
            if len(shapes) == 1:
                return TimeTensor(
                    data=np.stack(results, axis=0), dims=X.dims, channels=X.channels, y=X.y, metadata=X.metadata
                )
            return X.with_ragged(results)

        data = X.data
        b_ax = X.axis(Dim.BATCH) if X.has_dim(Dim.BATCH) else 0
        B = data.shape[b_ax]
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
        other = [i for i in range(data.ndim) if i not in (b_ax, ch_ax)]
        perm = [b_ax] + other + [ch_ax]
        transposed = np.transpose(data, perm)
        T_merged = int(np.prod(transposed.shape[1:-1]))
        C = transposed.shape[-1]
        flat = transposed.reshape(B, T_merged, C)
        results = [self._call_transform(flat[i], state) for i in range(B)]
        return X.with_data(np.stack(results, axis=0))

    def __repr__(self):
        return f"StatefulFunctionTransform({self.name}, scope={self.scope.name})"


# ======================================================================
# FeatureExtractor — dimensionality-changing extraction, single or multi-fn
# ======================================================================


class FeatureExtractor(Transform):
    """Extract features per channel, with single-function or multi-function support.

    Operates in two modes selected by which constructor argument is provided:

    **Single-fn mode** (``fn=...``)
        One callable that receives a 1-D vector (one channel of one sample)
        and may return any shape.  The consumed axis (``input_dim``) is
        replaced by ``output_dims`` in the schema.

        Use this for spectrograms, STFT, wavelets, and other transforms that
        produce multi-dimensional output per channel.

    **Multi-fn mode** (``functions=[...]``)
        A list of ``(name, fn)`` or ``(name, fn, per_fn_kwargs)`` entries.
        Each ``fn`` receives a 1-D vector and **must** return a scalar or a
        1-D array.  All outputs are concatenated along a new ``Dim.FEATURE``
        axis (replacing ``input_dim``), exactly like sklearn's FeatureUnion.

        Use this to replace ``StatisticalFeatures`` or ``FrameFeatureBank``
        with plain functions, or to build custom multi-feature pipelines.

    Parameters
    ----------
    fn : callable, optional
        *Single-fn mode only.*  ``fn(x: ndarray[T]) -> ndarray``

        - ``(T',)`` → 1-D output, axis length changes
        - ``(D1, D2, ...)`` → multi-dim, axis is replaced by new axes
        - ``()`` → scalar, axis is removed entirely

    input_dim : Dim
        Which axis the function(s) consume.  Default: ``Dim.TIME``.

    output_dims : tuple of Dim
        *Single-fn mode only.*  Axes that replace ``input_dim``:

        - ``(Dim.TIME,)`` — same axis, different length
        - ``(Dim.FREQUENCY, Dim.TIME)`` — e.g. spectrogram
        - ``()`` — scalar, axis removed

    fn_args : tuple
        Extra positional args forwarded to ``fn`` (single-fn mode).

    fn_kwargs : dict
        Extra keyword args forwarded to ``fn`` (single-fn mode).

    name : str or None
        Display name for single-fn mode (repr / debugging).

    functions : list of (name, fn) or (name, fn, kwargs), optional
        *Multi-fn mode only.*  Each entry is a 2-tuple ``(name, fn)`` or
        a 3-tuple ``(name, fn, kwargs_dict)``.  ``fn`` must return a scalar
        or 1-D array; 2-D+ output raises ``ValueError``.

    scope : Scope
        *Multi-fn mode only.*  Currently only ``Scope.PER_CHANNEL``
        (default) is supported — each function receives a 1-D array for
        one channel of one sample.

    Attributes
    ----------
    feature_names_out_ : list of str
        Set after the first ``transform`` call (multi-fn mode).  Names are
        ``"{fn_name}__{ch_name}"`` for scalar outputs and
        ``"{fn_name}__{k}__{ch_name}"`` for vector outputs.

    Notes
    -----
    Lambdas and locally-defined functions are not picklable, which means
    ``GridSearchCV(n_jobs > 1)`` will fail.  Use named functions or
    ``functools.partial`` if you need multiprocessing.

    Examples
    --------
    **Single-fn** — mel spectrogram::

        FeatureExtractor(
            fn=lambda x: librosa.feature.melspectrogram(y=x, sr=16000, n_mels=64),
            input_dim=Dim.TIME,
            output_dims=(Dim.FREQUENCY, Dim.TIME),
        )
        # (B, T, 1) → (B, 64, FRAMES, 1)
        # (B, K, T, 1) → (B, K, 64, FRAMES, 1)   windowed input works too

    **Single-fn** — scalar energy per channel::

        FeatureExtractor(fn=lambda x: np.sum(x**2), input_dim=Dim.TIME, output_dims=())
        # (B, T, C) → (B, C)

    **Multi-fn** — statistical summary (replaces StatisticalFeatures)::

        FeatureExtractor(
            functions=[
                ("mean",   np.mean),
                ("std",    np.std),
                ("energy", lambda x: np.sum(x**2)),
            ],
            input_dim=Dim.TIME,
        )
        # (B, T, C) → (B, 3, C)  with Dim.FEATURE axis
        # feature_names_out_: ["mean__ch0", "mean__ch1", ..., "std__ch0", ...]

    **Multi-fn** — mix scalar + vector (replaces FrameFeatureBank)::

        FeatureExtractor(
            functions=[
                ("zcr",  lambda x: np.mean(np.diff(np.signbit(x)))),  # scalar
                ("mfcc", compute_mfcc, {"n_mfcc": 13}),               # 13-elem
            ],
            input_dim=Dim.TIME,
        )
        # (B, K, T, C) → (B, K, 14, C)  with Dim.FEATURE axis
    """

    supports_ragged = True

    def __init__(
        self,
        fn: Callable | None = None,
        input_dim: Dim = Dim.TIME,
        output_dims: tuple[Dim, ...] = (Dim.TIME,),
        fn_args: tuple = (),
        fn_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        functions: list | None = None,
        scope: Scope = Scope.PER_CHANNEL,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.fn_args = fn_args
        self.fn_kwargs = {} if fn_kwargs is None else fn_kwargs
        self.name = name
        self.functions = functions
        self.scope = scope
        self._setup_mode()

    # ------------------------------------------------------------------
    # Mode detection and validation
    # ------------------------------------------------------------------

    def _setup_mode(self) -> None:
        """Validate parameters and derive internal state from public attrs."""
        fn, functions = self.fn, self.functions
        if fn is not None and functions is not None:
            raise ValueError("Provide either 'fn' (single-fn mode) or 'functions' (multi-fn mode), not both.")
        if fn is None and functions is None:
            raise ValueError("Provide either 'fn' (single-fn mode) or 'functions' (multi-fn mode).")
        if functions is not None:
            if len(functions) == 0:
                raise ValueError("'functions' must not be empty.")
            if self.output_dims != (Dim.TIME,):
                # output_dims was explicitly set alongside functions — that's an error
                # (We allow the default value through because users may not set it at all)
                raise ValueError(
                    "'output_dims' is only used in single-fn mode. "
                    "In multi-fn mode the output axis is always Dim.FEATURE."
                )
            if self.scope == Scope.PER_SAMPLE:
                raise NotImplementedError("Scope.PER_SAMPLE is not yet supported in multi-fn mode.")
            # Normalise to 3-tuples (name, fn, kwargs)
            normalised = []
            for entry in functions:
                if len(entry) == 2:
                    normalised.append((entry[0], entry[1], {}))
                elif len(entry) == 3:
                    normalised.append((entry[0], entry[1], entry[2] or {}))
                else:
                    raise ValueError(f"Each entry in 'functions' must be a 2- or 3-tuple, got length {len(entry)}.")
            self._functions_normalized: list[tuple[str, Callable, dict]] = normalised
            self._mode = "multi"
        else:
            self._functions_normalized = []
            self._mode = "single"
            # Ensure name is set for single-fn mode
            if self.name is None:
                self.name = getattr(fn, "__name__", "anonymous")

    # ------------------------------------------------------------------
    # sklearn compat
    # ------------------------------------------------------------------

    def set_params(self, **params: Any) -> "FeatureExtractor":
        super().set_params(**params)
        # Re-derive _mode and _functions_normalized after any parameter change.
        # Guard output_dims check: if only functions is being set (not output_dims),
        # reset output_dims to default first to avoid spurious conflict error.
        if "functions" in params and "output_dims" not in params:
            self.output_dims = (Dim.TIME,)
        self._setup_mode()
        return self

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "FeatureExtractor":
        self._is_fitted = True
        return self

    def _call_fn(self, arr: np.ndarray) -> np.ndarray:
        """Call the single fn (single-fn mode)."""
        return np.asarray(self.fn(arr, *self.fn_args, **self.fn_kwargs))

    def transform(self, X: TimeTensor) -> TimeTensor:
        if self._mode == "single":
            if X.is_ragged:
                return self._transform_ragged(X)
            return self._transform_dense(X)
        else:
            if X.is_ragged:
                return self._transform_multi_ragged(X)
            return self._transform_multi_dense(X)

    # ------------------------------------------------------------------
    # Single-fn mode — dense and ragged (unchanged from original)
    # ------------------------------------------------------------------

    def _transform_dense(self, X: TimeTensor) -> TimeTensor:
        data = X.data
        consumed_ax = X.axis(self.input_dim)
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        leading_axes = [i for i in range(data.ndim) if i not in (consumed_ax, ch_ax)]
        perm = leading_axes + [consumed_ax, ch_ax]
        transposed = np.transpose(data, perm)

        leading_shape = transposed.shape[: len(leading_axes)]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        N = int(np.prod(leading_shape)) if leading_shape else 1
        flat = transposed.reshape(N, T, C)

        probe = self._call_fn(flat[0, :, 0])
        out_shape = probe.shape

        full_shape = (N,) + out_shape + (C,)
        out = np.empty(full_shape, dtype=probe.dtype)
        out[0, ..., 0] = probe

        for n in range(N):
            for c in range(C):
                if n == 0 and c == 0:
                    continue
                out[n, ..., c] = self._call_fn(flat[n, :, c])

        final_shape = leading_shape + out_shape + (C,)
        out = out.reshape(final_shape)

        leading_dims = tuple(X.dims[i] for i in leading_axes)
        ch_d = ch_dim if ch_dim is not None else Dim.CHANNEL
        new_dims = leading_dims + self.output_dims + (ch_d,)

        return X.with_data(out, dims=new_dims, channels=X.channels)

    def _transform_ragged(self, X: TimeTensor) -> TimeTensor:
        new_samples = []
        for sample in X._samples:
            C = sample.shape[-1]
            ch_results = [self._call_fn(sample[:, c]) for c in range(C)]

            if len(self.output_dims) == 0:
                new_samples.append(np.array([r.item() for r in ch_results]))
            elif ch_results[0].ndim <= 1:
                cols = [r.ravel() if r.ndim == 0 else r for r in ch_results]
                new_samples.append(np.column_stack(cols))
            else:
                new_samples.append(np.stack(ch_results, axis=-1))

        ch_dim = X._find_channel_dim(X.schema)
        ch_d = ch_dim if ch_dim is not None else Dim.CHANNEL

        if len(self.output_dims) == 0:
            stacked = np.stack(new_samples, axis=0)
            new_dims = (Dim.BATCH, ch_d)
            return TimeTensor(
                data=stacked,
                dims=new_dims,
                channels=X.channels,
                y=X.y,
                metadata=X.metadata,
            )

        shapes = set(s.shape for s in new_samples)
        new_dims = (Dim.BATCH,) + self.output_dims + (ch_d,)
        if len(shapes) == 1:
            stacked = np.stack(new_samples, axis=0)
            return TimeTensor(
                data=stacked,
                dims=new_dims,
                channels=X.channels,
                y=X.y,
                metadata=X.metadata,
            )
        return X.with_ragged(new_samples, dims=new_dims, channels=X.channels)

    # ------------------------------------------------------------------
    # Multi-fn mode — dense
    # ------------------------------------------------------------------

    def _transform_multi_dense(self, X: TimeTensor) -> TimeTensor:
        data = X.data
        consumed_ax = X.axis(self.input_dim)
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
        ch_d = ch_dim if ch_dim is not None else Dim.CHANNEL

        leading_axes = [i for i in range(data.ndim) if i not in (consumed_ax, ch_ax)]
        perm = leading_axes + [consumed_ax, ch_ax]
        transposed = np.transpose(data, perm)

        leading_shape = transposed.shape[: len(leading_axes)]
        T = transposed.shape[-2]
        C = transposed.shape[-1]
        N = int(np.prod(leading_shape)) if leading_shape else 1
        flat = transposed.reshape(N, T, C)

        fn_outputs: list[np.ndarray] = []  # each is (N, K_i, C)
        feature_names: list[str] = []  # one name per feature (K_total total)

        for fname, fn, kwargs in self._functions_normalized:
            probe = np.asarray(fn(flat[0, :, 0], **kwargs)).ravel()
            K_i = probe.size
            if np.asarray(fn(flat[0, :, 0], **kwargs)).ndim > 1:
                raise ValueError(
                    f"In multi-fn mode each function must return a scalar or 1-D array. "
                    f"Function '{fname}' returned shape "
                    f"{np.asarray(fn(flat[0, :, 0], **kwargs)).shape}."
                )
            fn_out = np.empty((N, K_i, C), dtype=probe.dtype)
            fn_out[0, :, 0] = probe
            for n in range(N):
                for c in range(C):
                    if n == 0 and c == 0:
                        continue
                    fn_out[n, :, c] = np.asarray(fn(flat[n, :, c], **kwargs)).ravel()
            fn_outputs.append(fn_out)

            # Feature names along the FEATURE axis (one per K_i, not per channel)
            for k in range(K_i):
                feature_names.append(fname if K_i == 1 else f"{fname}__{k}")

        combined = np.concatenate(fn_outputs, axis=1)  # (N, K_total, C)
        K_total = combined.shape[1]
        out = combined.reshape(*leading_shape, K_total, C)

        leading_dims = tuple(X.dims[i] for i in leading_axes)
        new_dims = leading_dims + (Dim.FEATURE, ch_d)

        self.feature_names_out_ = feature_names
        # Channels along CHANNEL axis stay the same as input
        meta = {**X.metadata, "_feature_names": feature_names}
        return X.with_data(out, dims=new_dims, channels=X.channels, metadata=meta)

    # ------------------------------------------------------------------
    # Multi-fn mode — ragged
    # ------------------------------------------------------------------

    def _transform_multi_ragged(self, X: TimeTensor) -> TimeTensor:
        ch_dim = X._find_channel_dim(X.schema)
        ch_d = ch_dim if ch_dim is not None else Dim.CHANNEL

        # Use first sample to probe output sizes and build feature names
        first = X._samples[0]

        fn_sizes: list[int] = []
        feature_names: list[str] = []  # one name per feature (K_total total)
        for fname, fn, kwargs in self._functions_normalized:
            probe = np.asarray(fn(first[:, 0], **kwargs)).ravel()
            K_i = probe.size
            if np.asarray(fn(first[:, 0], **kwargs)).ndim > 1:
                raise ValueError(
                    f"In multi-fn mode each function must return a scalar or 1-D array. "
                    f"Function '{fname}' returned a multi-dimensional array."
                )
            fn_sizes.append(K_i)
            for k in range(K_i):
                feature_names.append(fname if K_i == 1 else f"{fname}__{k}")

        new_samples = []
        for sample in X._samples:
            C_s = sample.shape[-1]
            fn_outputs = []
            for (fname, fn, kwargs), K_i in zip(self._functions_normalized, fn_sizes):
                fn_out = np.empty((K_i, C_s))
                for c in range(C_s):
                    fn_out[:, c] = np.asarray(fn(sample[:, c], **kwargs)).ravel()
                fn_outputs.append(fn_out)
            new_samples.append(np.concatenate(fn_outputs, axis=0))  # (K_total, C)

        stacked = np.stack(new_samples, axis=0)  # (B, K_total, C)
        new_dims = (Dim.BATCH, Dim.FEATURE, ch_d)

        self.feature_names_out_ = feature_names
        # Channels along CHANNEL axis stay the same as input
        meta = {**X.metadata, "_feature_names": feature_names}
        return TimeTensor(
            data=stacked,
            dims=new_dims,
            channels=X.channels,
            y=X.y,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._mode == "single":
            out = ", ".join(d.name for d in self.output_dims) or "(scalar)"
            return f"FeatureExtractor(fn={self.name}, {self.input_dim.name} → [{out}])"
        fn_names = ", ".join(n for n, *_ in self.functions)
        return f"FeatureExtractor(functions=[{fn_names}], {self.input_dim.name} → FEATURE)"
