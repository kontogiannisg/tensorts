"""
Transform base classes and composition primitives.

All transforms follow the sklearn fit/transform protocol and operate on
TimeTensor objects.  Transforms declare which dims they expect and produce,
and use semantic axis resolution — never hardcoded integer indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim, DimSchema
from ..core.tensor import TimeTensor

# ======================================================================
# Scope enum
# ======================================================================


class Scope(Enum):
    """Declares how a transform relates to the tensor axes."""

    GLOBAL = auto()  # Needs all axes together (e.g., PCA)
    PER_CHANNEL = auto()  # Independent per channel, vectorized over batch+time
    PER_SAMPLE = auto()  # Independent per sample, vectorized over channels
    ELEMENTWISE = auto()  # Independent per element


# ======================================================================
# Base Transform
# ======================================================================


class Transform(ABC):
    """Base class for all transforms.

    Subclasses implement ``fit`` and ``transform``.  The base class provides
    ``fit_transform``, fitted-state tracking, and ragged dispatch.

    Class attributes
    ----------------
    input_dims : tuple of Dim or None
        Expected input dim schema.  ``None`` means any schema is accepted.
    output_dims : tuple of Dim or None
        Dim schema of the output.  ``None`` means same as input.
    supports_ragged : bool
        Whether this transform can handle ragged TimeTensors directly.
    scope : Scope
        Operational scope (for documentation / routing; not enforced).
    """

    input_dims: tuple[Dim, ...] | None = None
    output_dims: tuple[Dim, ...] | None = None
    supports_ragged: bool = False
    supports_partial_fit: bool = False
    scope: Scope = Scope.GLOBAL

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} has not been fitted. Call fit() first.")

    def _validate_input(self, X: TimeTensor) -> None:
        if self.input_dims is not None and not X.is_ragged:
            expected = DimSchema(self.input_dims)
            if X.schema != expected:
                raise ValueError(f"{self.__class__.__name__} expects dims {expected}, got {X.schema}.")
        if X.is_ragged and not self.supports_ragged:
            raise ValueError(
                f"{self.__class__.__name__} does not support ragged input. Apply windowing or padding first."
            )

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Transform":
        """Fit the transform on training data. Default: no-op (stateless)."""
        self._validate_input(X)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: TimeTensor) -> TimeTensor:
        """Apply the transform.  Must be implemented by subclasses."""
        ...

    def fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def partial_fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Transform":
        """Incremental fit on a batch of data.

        Default falls back to ``fit()`` (full refit).  Subclasses that
        support true incremental learning override this and set
        ``supports_partial_fit = True``.
        """
        return self.fit(X, y)

    def partial_fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        """Incremental fit and transform in one step."""
        return self.partial_fit(X, y).transform(X)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters (sklearn-compatible)."""
        import inspect

        sig = inspect.signature(self.__class__.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                params[name] = getattr(self, f"_{name}")
        return params

    def set_params(self, **params) -> "Transform":
        """Set parameters — enables sklearn GridSearchCV compatibility."""
        valid = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for {self.__class__.__name__}. "
                    f"Valid parameters: {sorted(valid.keys())}."
                )
            setattr(self, key, value)
        return self


# ======================================================================
# Pipeline
# ======================================================================


class Pipeline(Transform):
    """Sequential composition of named transforms.

    Follows sklearn Pipeline semantics: ``fit`` calls ``fit_transform`` on
    all but the last step, then ``fit`` on the last.  ``transform`` calls
    ``transform`` on every step.

    Parameters
    ----------
    steps : list of (name, Transform) tuples
    """

    supports_ragged = True  # delegates to individual steps
    supports_partial_fit = True  # delegates to individual steps

    def __init__(self, steps: list[tuple[str, Transform]]) -> None:
        super().__init__()
        self._validate_steps(steps)
        self.steps = steps

    @staticmethod
    def _validate_steps(steps: list[tuple[str, Transform]]) -> None:
        names = set()
        for name, transform in steps:
            if not isinstance(name, str):
                raise TypeError(f"Step name must be str, got {type(name)}.")
            if not isinstance(transform, Transform):
                raise TypeError(f"Step '{name}' is not a Transform: {type(transform)}.")
            if name in names:
                raise ValueError(f"Duplicate step name: {name!r}.")
            names.add(name)

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Pipeline":
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y)
            # If labels were updated by the transform (e.g. windowing), propagate
            if Xt.y is not None:
                y = Xt.y
        # Fit (but don't transform) the last step
        last_name, last_step = self.steps[-1]
        last_step.fit(Xt, y)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        Xt = X
        for name, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        Xt = X
        for name, step in self.steps:
            Xt = step.fit_transform(Xt, y)
            if Xt.y is not None:
                y = Xt.y
        self._is_fitted = True
        return Xt

    def partial_fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Pipeline":
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.partial_fit_transform(Xt, y)
            if Xt.y is not None:
                y = Xt.y
        last_name, last_step = self.steps[-1]
        last_step.partial_fit(Xt, y)
        self._is_fitted = True
        return self

    def partial_fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        Xt = X
        for name, step in self.steps:
            Xt = step.partial_fit_transform(Xt, y)
            if Xt.y is not None:
                y = Xt.y
        self._is_fitted = True
        return Xt

    def __getitem__(self, key: str | int) -> Transform:
        if isinstance(key, str):
            for name, step in self.steps:
                if name == key:
                    return step
            raise KeyError(f"Step {key!r} not found.")
        return self.steps[key][1]

    @property
    def named_steps(self) -> dict[str, Transform]:
        """Dict of step name → transform (sklearn-style access)."""
        return dict(self.steps)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"steps": self.steps}
        if deep:
            for name, step in self.steps:
                params[name] = step
                for key, val in step.get_params(deep=True).items():
                    params[f"{name}__{key}"] = val
        return params

    def set_params(self, **params) -> "Pipeline":
        """Set parameters using sklearn nested 'step__param' format."""
        for key, value in params.items():
            if "__" in key:
                step_name, param_rest = key.split("__", 1)
                self[step_name].set_params(**{param_rest: value})
            elif key == "steps":
                self._validate_steps(value)
                self.steps = value
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for Pipeline. Use 'steps' or 'step_name__param_name' format."
                )
        return self


# ======================================================================
# ChannelTransform
# ======================================================================


class ChannelTransform(Transform):
    """Apply different transforms to different channel groups.

    Parameters
    ----------
    transforms : dict mapping group name / glob → Transform
        Each key is resolved via the ChannelIndex (group name or glob pattern).
        Each value is a Transform applied independently to those channels.
    remainder : str
        What to do with channels not matched by any key:
        - ``'passthrough'``: include them unchanged
        - ``'drop'``: exclude them from output
    """

    supports_ragged = True
    scope = Scope.PER_CHANNEL

    def __init__(
        self,
        transforms: dict[str, Transform],
        remainder: str = "passthrough",
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.remainder = remainder
        self._resolved: list[tuple[str, np.ndarray, Transform]] | None = None
        self._remainder_indices: np.ndarray | None = None

    def _resolve_groups(self, X: TimeTensor) -> None:
        """Resolve group keys to integer indices."""
        ci = X.channels
        if ci is None:
            raise RuntimeError("ChannelTransform requires named channels.")

        all_matched = set()
        self._resolved = []
        for key, tfm in self.transforms.items():
            # Try group first, then glob/name
            try:
                indices = ci.group(key)
            except KeyError:
                indices = ci.resolve(key)
            self._resolved.append((key, indices, tfm))
            all_matched.update(indices.tolist())

        all_indices = set(range(len(ci)))
        remaining = sorted(all_indices - all_matched)
        self._remainder_indices = np.array(remaining) if remaining else None

    supports_partial_fit = True  # delegates to inner transforms

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelTransform":
        self._resolve_groups(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1

        for key, indices, tfm in self._resolved:
            sub = self._select_channels(X, indices, ch_axis)
            tfm.fit(sub, y)

        self._is_fitted = True
        return self

    def partial_fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelTransform":
        if self._resolved is None:
            self._resolve_groups(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1

        for key, indices, tfm in self._resolved:
            sub = self._select_channels(X, indices, ch_axis)
            tfm.partial_fit(sub, y)

        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if self._resolved is None:
            self._resolve_groups(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1

        parts = []
        channel_names = []
        for key, indices, tfm in self._resolved:
            sub = self._select_channels(X, indices, ch_axis)
            transformed = tfm.transform(sub)
            parts.append(transformed)
            if transformed.channels is not None:
                channel_names.extend(transformed.channels.names)

        if self.remainder == "passthrough" and self._remainder_indices is not None:
            remainder_sub = self._select_channels(X, self._remainder_indices, ch_axis)
            parts.append(remainder_sub)
            if remainder_sub.channels is not None:
                channel_names.extend(remainder_sub.channels.names)

        return self._concat_channels(parts, X, channel_names)

    @staticmethod
    def _select_channels(X: TimeTensor, indices: np.ndarray, ch_axis: int) -> TimeTensor:
        if X.is_ragged:
            new_samples = [s[..., indices] for s in X._samples]
            return X.with_ragged(
                new_samples,
                channels=X.channels.slice(indices) if X.channels else None,
            )
        new_data = np.take(X.data, indices, axis=ch_axis)
        return X.with_data(
            new_data,
            channels=X.channels.slice(indices) if X.channels else None,
        )

    @staticmethod
    def _concat_channels(
        parts: list[TimeTensor],
        original: TimeTensor,
        channel_names: list[str],
    ) -> TimeTensor:
        if parts[0].is_ragged:
            n_samples = len(parts[0]._samples)
            merged = [np.concatenate([p._samples[i] for p in parts], axis=-1) for i in range(n_samples)]
            return original.with_ragged(
                merged,
                channels=ChannelIndex(channel_names) if channel_names else None,
            )
        ch_axis = parts[0].axis(Dim.CHANNEL)
        merged = np.concatenate([p.data for p in parts], axis=ch_axis)
        return original.with_data(
            merged,
            channels=ChannelIndex(channel_names) if channel_names else None,
        )


# ======================================================================
# ChannelSelect
# ======================================================================


class ChannelSelect(Transform):
    """Apply a transform to selected channels; others pass through unchanged.

    Parameters
    ----------
    channels : str or list of str
        Glob pattern, group name, or explicit list of channel names.
    transform : Transform
        Transform to apply to the selected channels.
    """

    supports_ragged = True

    def __init__(self, channels: str | list[str], transform: Transform) -> None:
        super().__init__()
        self.channel_key = channels
        self.inner = transform  # avoid shadowing inherited .transform() method
        self._selected_idx: np.ndarray | None = None
        self._other_idx: np.ndarray | None = None

    def _resolve(self, X: TimeTensor) -> None:
        ci = X.channels
        if ci is None:
            raise RuntimeError("ChannelSelect requires named channels.")
        try:
            self._selected_idx = (
                ci.group(self.channel_key) if isinstance(self.channel_key, str) else ci.resolve(self.channel_key)
            )
        except KeyError:
            self._selected_idx = ci.resolve(self.channel_key)
        all_idx = set(range(len(ci)))
        other = sorted(all_idx - set(self._selected_idx.tolist()))
        self._other_idx = np.array(other) if other else None

    supports_partial_fit = True  # delegates to inner transform

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelSelect":
        self._resolve(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1
        sub = ChannelTransform._select_channels(X, self._selected_idx, ch_axis)
        self.inner.fit(sub, y)
        self._is_fitted = True
        return self

    def partial_fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelSelect":
        if self._selected_idx is None:
            self._resolve(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1
        sub = ChannelTransform._select_channels(X, self._selected_idx, ch_axis)
        self.inner.partial_fit(sub, y)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if self._selected_idx is None:
            self._resolve(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1

        sub = ChannelTransform._select_channels(X, self._selected_idx, ch_axis)
        transformed = self.inner.transform(sub)

        parts = []
        channel_names = []

        if transformed.channels:
            channel_names.extend(transformed.channels.names)
        parts.append(transformed)

        if self._other_idx is not None and len(self._other_idx) > 0:
            other = ChannelTransform._select_channels(X, self._other_idx, ch_axis)
            parts.append(other)
            if other.channels:
                channel_names.extend(other.channels.names)

        return ChannelTransform._concat_channels(parts, X, channel_names)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"channels": self.channel_key, "transform": self.inner}
        if deep:
            for k, v in self.inner.get_params(deep=True).items():
                params[f"transform__{k}"] = v
        return params

    def set_params(self, **params) -> "ChannelSelect":
        for key, value in params.items():
            if key == "channels":
                self.channel_key = value
            elif key == "transform":
                self.inner = value
            elif key.startswith("transform__"):
                self.inner.set_params(**{key[len("transform__") :]: value})
            else:
                raise ValueError(f"Invalid parameter {key!r} for ChannelSelect.")
        return self


# ======================================================================
# FeatureUnion
# ======================================================================


class FeatureUnion(Transform):
    """Run multiple transforms in parallel and concatenate their outputs.

    Each sub-transform receives the **same** input ``TimeTensor``.  Their
    outputs are flattened to 2-D ``(B, K_i)`` and horizontally concatenated
    into a single feature array with ``Dim.FEATURE`` as the last axis.

    When all sub-transforms share leading dims (e.g., ``BATCH, WINDOW``),
    those are preserved and only trailing dims are flattened.  This allows
    FeatureUnion to work naturally after ``SlidingWindow(flatten_batch=False)``
    for temporal DL pipelines.

    This is the tensorts equivalent of sklearn's ``FeatureUnion`` and is
    designed for combining extractors with different scopes (e.g.,
    per-channel statistics alongside per-sample cross-channel features).

    Parameters
    ----------
    transformers : list of (name, Transform) tuples
        Named sub-transforms to run in parallel.
    """

    supports_ragged = True
    supports_partial_fit = True

    def __init__(self, transformers: list[tuple[str, Transform]]) -> None:
        super().__init__()
        self._validate_transformers(transformers)
        self.transformers = transformers

    @staticmethod
    def _validate_transformers(transformers: list[tuple[str, Transform]]) -> None:
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = set()
        for name, tfm in transformers:
            if not isinstance(name, str):
                raise TypeError(f"Transformer name must be str, got {type(name)}.")
            if not isinstance(tfm, Transform):
                raise TypeError(f"Transformer '{name}' is not a Transform: {type(tfm)}.")
            if name in names:
                raise ValueError(f"Duplicate transformer name: {name!r}.")
            names.add(name)

    # Dims that are "trailing" — produced by feature extraction / channel axes.
    # Everything before the first occurrence of one of these is a "leading" dim
    # that FeatureUnion preserves (e.g., BATCH, WINDOW).
    _TRAILING_DIMS = frozenset({Dim.FEATURE, Dim.CHANNEL, Dim.FREQUENCY})

    @staticmethod
    def _split_leading(result: TimeTensor) -> tuple[tuple[int, ...], tuple[Dim, ...]]:
        """Find the boundary between leading (preserved) and trailing (flattened) dims.

        Returns ``(leading_shape, leading_dims)``.  The trailing dims are
        everything after ``leading_dims`` and will be collapsed into FEATURE.

        When no trailing dim is found (e.g., scalar PER_SAMPLE output with
        only BATCH or BATCH+WINDOW), all dims are treated as leading and the
        caller will expand a trailing size-1 feature axis.
        """
        dims = result.dims
        split = len(dims)  # default: everything is leading
        for i, d in enumerate(dims):
            if d in FeatureUnion._TRAILING_DIMS:
                split = i
                break
        leading_dims = dims[:split]
        leading_shape = result.data.shape[:split]
        return leading_shape, leading_dims

    @staticmethod
    def _flatten_trailing(
        result: TimeTensor,
    ) -> tuple[np.ndarray, list[str], tuple[int, ...], tuple[Dim, ...]]:
        """Flatten trailing (feature/channel) dims, preserve leading dims.

        Returns ``(flat, names, leading_shape, leading_dims)`` where
        ``flat`` has shape ``leading_shape + (K,)``.
        """
        if result.is_ragged:
            raise ValueError(
                "FeatureUnion requires all sub-transforms to produce dense output. "
                "Apply windowing or padding before using FeatureUnion with ragged transforms."
            )
        leading_shape, leading_dims = FeatureUnion._split_leading(result)
        data = result.data
        if leading_shape == data.shape:
            # No trailing dims (scalar output) — expand a size-1 feature axis
            flat = data.reshape(leading_shape + (1,))
        else:
            # Reshape to (*leading_shape, K) where K = product of trailing dims
            flat = data.reshape(leading_shape + (-1,))
        K = flat.shape[-1]

        # Build feature names for the K trailing features
        feature_names = result.metadata.get("_feature_names", None)
        channels = result.channels

        if feature_names is not None and channels is not None:
            if len(feature_names) * len(channels.names) == K:
                # Multi-fn per-channel: cross-product feature × channel
                names = [f"{fn}__{ch}" for fn in feature_names for ch in channels.names]
            elif len(feature_names) == K:
                names = list(feature_names)
            elif len(channels.names) == K:
                names = list(channels.names)
            else:
                names = [f"f_{i}" for i in range(K)]
        elif feature_names is not None:
            if len(feature_names) == K:
                names = list(feature_names)
            else:
                names = [f"f_{i}" for i in range(K)]
        elif channels is not None:
            if len(channels.names) == K:
                names = list(channels.names)
            else:
                n_per_ch = K // len(channels.names) if len(channels.names) > 0 else K
                names = [f"f{j}__{ch}" for j in range(n_per_ch) for ch in channels.names]
        else:
            names = [f"f_{i}" for i in range(K)]

        # Safety: if name count doesn't match width, fall back
        if len(names) != K:
            names = [f"f_{i}" for i in range(K)]

        return flat, names, leading_shape, leading_dims

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "FeatureUnion":
        for name, tfm in self.transformers:
            tfm.fit(X, y)
        self._is_fitted = True
        return self

    @staticmethod
    def _validate_leading(
        entries: list[tuple[str, tuple[int, ...], tuple[Dim, ...]]],
    ) -> None:
        """Ensure all sub-transforms agree on leading shape and dims."""
        first_name, first_shape, first_dims = entries[0]
        for tname, lshape, ldims in entries[1:]:
            if ldims != first_dims or lshape != first_shape:
                raise ValueError(
                    f"FeatureUnion sub-transforms have incompatible leading shapes: "
                    f"'{first_name}' produced {first_dims} with shape {first_shape}, "
                    f"but '{tname}' produced {ldims} with shape {lshape}. "
                    f"All sub-transforms must share the same leading dimensions."
                )

    def _assemble(
        self,
        parts: list[np.ndarray],
        all_names: list[str],
        leading_dims: tuple[Dim, ...],
        X: TimeTensor,
    ) -> TimeTensor:
        """Concatenate sub-transform outputs and build the output TimeTensor."""
        combined = np.concatenate(parts, axis=-1)
        out_dims = leading_dims + (Dim.FEATURE,)
        meta = {**X.metadata, "_feature_names": all_names}
        return TimeTensor(
            data=combined,
            dims=out_dims,
            channels=all_names,
            y=X.y,
            metadata=meta,
        )

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        parts: list[np.ndarray] = []
        all_names: list[str] = []
        leading_info: list[tuple[str, tuple[int, ...], tuple[Dim, ...]]] = []

        for name, tfm in self.transformers:
            result = tfm.transform(X)
            flat, names, lshape, ldims = self._flatten_trailing(result)
            parts.append(flat)
            all_names.extend(names)
            leading_info.append((name, lshape, ldims))

        self._validate_leading(leading_info)
        return self._assemble(parts, all_names, leading_info[0][2], X)

    def fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        parts: list[np.ndarray] = []
        all_names: list[str] = []
        leading_info: list[tuple[str, tuple[int, ...], tuple[Dim, ...]]] = []

        for name, tfm in self.transformers:
            result = tfm.fit_transform(X, y)
            flat, names, lshape, ldims = self._flatten_trailing(result)
            parts.append(flat)
            all_names.extend(names)
            leading_info.append((name, lshape, ldims))

        self._validate_leading(leading_info)
        self._is_fitted = True
        return self._assemble(parts, all_names, leading_info[0][2], X)

    def partial_fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "FeatureUnion":
        for name, tfm in self.transformers:
            tfm.partial_fit(X, y)
        self._is_fitted = True
        return self

    def partial_fit_transform(self, X: TimeTensor, y: np.ndarray | None = None) -> TimeTensor:
        parts: list[np.ndarray] = []
        all_names: list[str] = []
        leading_info: list[tuple[str, tuple[int, ...], tuple[Dim, ...]]] = []

        for name, tfm in self.transformers:
            result = tfm.partial_fit_transform(X, y)
            flat, names, lshape, ldims = self._flatten_trailing(result)
            parts.append(flat)
            all_names.extend(names)
            leading_info.append((name, lshape, ldims))

        self._validate_leading(leading_info)
        self._is_fitted = True
        return self._assemble(parts, all_names, leading_info[0][2], X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"transformers": self.transformers}
        if deep:
            for name, tfm in self.transformers:
                params[name] = tfm
                for key, val in tfm.get_params(deep=True).items():
                    params[f"{name}__{key}"] = val
        return params

    def set_params(self, **params) -> "FeatureUnion":
        for key, value in params.items():
            if "__" in key:
                tfm_name, param_rest = key.split("__", 1)
                for name, tfm in self.transformers:
                    if name == tfm_name:
                        tfm.set_params(**{param_rest: value})
                        break
                else:
                    raise ValueError(f"Unknown transformer {tfm_name!r} in FeatureUnion.")
            elif key == "transformers":
                self._validate_transformers(value)
                self.transformers = value
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for FeatureUnion. Use 'transformers' or 'name__param' format."
                )
        return self

    def __repr__(self) -> str:
        names = ", ".join(n for n, _ in self.transformers)
        return f"FeatureUnion([{names}])"
