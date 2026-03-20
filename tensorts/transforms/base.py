"""
Transform base classes and composition primitives.

All transforms follow the sklearn fit/transform protocol and operate on
TimeTensor objects.  Transforms declare which dims they expect and produce,
and use semantic axis resolution — never hardcoded integer indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Sequence

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim, DimSchema
from ..core.tensor import TimeTensor


# ======================================================================
# Scope enum
# ======================================================================

class Scope(Enum):
    """Declares how a transform relates to the tensor axes."""
    GLOBAL = auto()       # Needs all axes together (e.g., PCA)
    PER_CHANNEL = auto()  # Independent per channel, vectorized over batch+time
    PER_SAMPLE = auto()   # Independent per sample, vectorized over channels
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
    scope: Scope = Scope.GLOBAL

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call fit() first."
            )

    def _validate_input(self, X: TimeTensor) -> None:
        if self.input_dims is not None and not X.is_ragged:
            expected = DimSchema(self.input_dims)
            if X.schema != expected:
                raise ValueError(
                    f"{self.__class__.__name__} expects dims {expected}, "
                    f"got {X.schema}."
                )
        if X.is_ragged and not self.supports_ragged:
            raise ValueError(
                f"{self.__class__.__name__} does not support ragged input. "
                "Apply windowing or padding first."
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

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.get_params().items()
        )
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
                raise TypeError(
                    f"Step '{name}' is not a Transform: {type(transform)}."
                )
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
                    f"Invalid parameter {key!r} for Pipeline. "
                    "Use 'steps' or 'step_name__param_name' format."
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

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelTransform":
        self._resolve_groups(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1

        for key, indices, tfm in self._resolved:
            sub = self._select_channels(X, indices, ch_axis)
            tfm.fit(sub, y)

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
    def _select_channels(
        X: TimeTensor, indices: np.ndarray, ch_axis: int
    ) -> TimeTensor:
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
            merged = [
                np.concatenate([p._samples[i] for p in parts], axis=-1)
                for i in range(n_samples)
            ]
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
            self._selected_idx = ci.group(self.channel_key) if isinstance(self.channel_key, str) else ci.resolve(self.channel_key)
        except KeyError:
            self._selected_idx = ci.resolve(self.channel_key)
        all_idx = set(range(len(ci)))
        other = sorted(all_idx - set(self._selected_idx.tolist()))
        self._other_idx = np.array(other) if other else None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ChannelSelect":
        self._resolve(X)
        ch_axis = X.axis(Dim.CHANNEL) if not X.is_ragged else -1
        sub = ChannelTransform._select_channels(X, self._selected_idx, ch_axis)
        self.inner.fit(sub, y)
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
                self.inner.set_params(**{key[len("transform__"):]: value})
            else:
                raise ValueError(f"Invalid parameter {key!r} for ChannelSelect.")
        return self
