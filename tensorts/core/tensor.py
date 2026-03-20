"""
TimeTensor — the central data container for tensorts.

A NumPy array where every axis has a semantic role, the channel axis carries
named metadata, and an optional label vector propagates through transforms.

Supports two modes:
- **Dense**: regular ndarray, all samples same length along TIME.
- **Ragged**: list of arrays, each sample may have different TIME length.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from .channels import ChannelIndex
from .dims import Dim, DimSchema


class _ChannelAccessor:
    """Syntactic sugar for ``ts.ch['accel_*']`` style access."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: "TimeTensor") -> None:
        self._tensor = tensor

    def __getitem__(self, key) -> "TimeTensor":
        ts = self._tensor
        if ts._channels is None:
            raise RuntimeError("Cannot index channels — no ChannelIndex defined.")
        indices = ts._channels.resolve(key)

        if ts.is_ragged:
            new_samples = [s[..., indices] for s in ts._samples]
            return TimeTensor._from_ragged_internal(
                samples=new_samples,
                schema=ts._schema,
                channels=ts._channels.slice(indices),
                y=ts._y,
                metadata=ts._metadata,
            )
        else:
            ch_axis = ts.axis(Dim.CHANNEL)
            new_data = np.take(ts._data, indices, axis=ch_axis)
            return ts.with_data(new_data, channels=ts._channels.slice(indices))

    def group(self, name: str) -> "TimeTensor":
        ts = self._tensor
        if ts._channels is None:
            raise RuntimeError("Cannot index channels — no ChannelIndex defined.")
        indices = ts._channels.group(name)
        if ts.is_ragged:
            new_samples = [s[..., indices] for s in ts._samples]
            return TimeTensor._from_ragged_internal(
                samples=new_samples,
                schema=ts._schema,
                channels=ts._channels.slice(indices),
                y=ts._y,
                metadata=ts._metadata,
            )
        else:
            ch_axis = ts.axis(Dim.CHANNEL)
            new_data = np.take(ts._data, indices, axis=ch_axis)
            return ts.with_data(new_data, channels=ts._channels.slice(indices))


class TimeTensor:
    """A semantically-annotated tensor for multivariate time series.

    Parameters
    ----------
    data : ndarray
        Dense array of shape matching *dims*.
    dims : tuple of Dim
        Semantic role for each axis, e.g. ``(Dim.BATCH, Dim.TIME, Dim.CHANNEL)``.
    channels : list of str, ChannelIndex, or None
        Names for the CHANNEL (or FEATURE) axis. If *None* and data has a
        CHANNEL dim, an auto-generated index is created.
    channel_groups : dict, optional
        Shorthand for defining groups at creation time.
    y : array-like or None
        Labels aligned with the BATCH axis.
    metadata : dict, optional
        Arbitrary metadata (sampling_rate, units, provenance, etc.).
    """

    __slots__ = (
        "_data",
        "_samples",
        "_schema",
        "_channels",
        "_y",
        "_metadata",
    )

    def __init__(
        self,
        data: np.ndarray,
        dims: tuple[Dim, ...] | Sequence[Dim],
        channels: list[str] | ChannelIndex | None = None,
        channel_groups: dict[str, list[str] | str] | None = None,
        y: np.ndarray | Sequence | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        data = np.asarray(data)
        schema = DimSchema(dims)

        if data.ndim != schema.ndim:
            raise ValueError(
                f"Data has {data.ndim} axes but dims specifies {schema.ndim}: {schema}"
            )

        # Build or validate ChannelIndex
        ch_dim = self._find_channel_dim(schema)
        if ch_dim is not None:
            ch_size = data.shape[schema.axis(ch_dim)]
            if channels is None:
                channels = ChannelIndex.auto(ch_size)
            elif isinstance(channels, (list, tuple)):
                channels = ChannelIndex(channels, groups=channel_groups)
            if len(channels) != ch_size:
                raise ValueError(
                    f"ChannelIndex has {len(channels)} names but axis "
                    f"{ch_dim.name} has size {ch_size}."
                )
            if channel_groups and isinstance(channels, ChannelIndex):
                for gname, pattern in channel_groups.items():
                    if gname not in channels.groups:
                        channels.add_group(gname, pattern)
        elif channels is not None:
            if isinstance(channels, (list, tuple)):
                channels = ChannelIndex(channels, groups=channel_groups)

        # Labels
        if y is not None:
            y = np.asarray(y)
            if schema.has(Dim.BATCH) and len(y) != data.shape[schema.axis(Dim.BATCH)]:
                raise ValueError(
                    f"Label length {len(y)} doesn't match batch size "
                    f"{data.shape[schema.axis(Dim.BATCH)]}."
                )

        self._data = data
        self._samples = None  # not ragged
        self._schema = schema
        self._channels = channels
        self._y = y
        self._metadata = metadata or {}

    @staticmethod
    def _find_channel_dim(schema: DimSchema) -> Dim | None:
        """Find the CHANNEL or FEATURE dim if present."""
        for d in (Dim.CHANNEL, Dim.FEATURE):
            if schema.has(d):
                return d
        return None

    # ------------------------------------------------------------------
    # Ragged constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ragged(
        cls,
        samples: list[np.ndarray],
        channels: list[str] | ChannelIndex | None = None,
        channel_groups: dict[str, list[str] | str] | None = None,
        y: np.ndarray | Sequence | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TimeTensor":
        """Create a ragged TimeTensor from variable-length samples.

        Each element of *samples* should be shaped ``(T_i, C)`` or ``(T_i,)``.
        """
        if not samples:
            raise ValueError("Cannot create TimeTensor from empty sample list.")

        ndim_0 = samples[0].ndim
        for i, s in enumerate(samples):
            s = np.asarray(s)
            samples[i] = s
            if s.ndim != ndim_0:
                raise ValueError(
                    f"Sample {i} has {s.ndim} dims, expected {ndim_0}."
                )
            if ndim_0 >= 2 and s.shape[1:] != samples[0].shape[1:]:
                raise ValueError(
                    f"Sample {i} has shape {s.shape}, expected (T_i, {samples[0].shape[1:]})."
                )

        # Promote 1-D samples to (T_i, 1)
        if ndim_0 == 1:
            samples = [s[:, np.newaxis] for s in samples]

        n_channels = samples[0].shape[-1]
        dims = (Dim.BATCH, Dim.TIME, Dim.CHANNEL)
        schema = DimSchema(dims)

        if channels is None:
            channels = ChannelIndex.auto(n_channels)
        elif isinstance(channels, (list, tuple)):
            channels = ChannelIndex(channels, groups=channel_groups)

        if channel_groups and isinstance(channels, ChannelIndex):
            for gname, pattern in channel_groups.items():
                if gname not in channels.groups:
                    channels.add_group(gname, pattern)

        if len(channels) != n_channels:
            raise ValueError(
                f"ChannelIndex has {len(channels)} names but samples have {n_channels} channels."
            )

        if y is not None:
            y = np.asarray(y)
            if len(y) != len(samples):
                raise ValueError(
                    f"Label length {len(y)} doesn't match sample count {len(samples)}."
                )

        return cls._from_ragged_internal(
            samples=samples,
            schema=schema,
            channels=channels,
            y=y,
            metadata=metadata or {},
        )

    @classmethod
    def _from_ragged_internal(
        cls,
        samples: list[np.ndarray],
        schema: DimSchema,
        channels: ChannelIndex | None,
        y: np.ndarray | None,
        metadata: dict,
    ) -> "TimeTensor":
        obj = object.__new__(cls)
        obj._data = None
        obj._samples = samples
        obj._schema = schema
        obj._channels = channels
        obj._y = y
        obj._metadata = metadata
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        if self.is_ragged:
            raise RuntimeError(
                "Cannot access .data on a ragged TimeTensor. "
                "Use .samples or apply windowing first."
            )
        return self._data

    @property
    def samples(self) -> list[np.ndarray]:
        """Individual samples (ragged mode) or slices along batch (dense)."""
        if self.is_ragged:
            return self._samples
        b_ax = self._schema.axis(Dim.BATCH)
        return [np.take(self._data, i, axis=b_ax) for i in range(self._data.shape[b_ax])]

    @property
    def is_ragged(self) -> bool:
        return self._samples is not None

    @property
    def lengths(self) -> list[int] | None:
        """Per-sample time-axis lengths (ragged mode only)."""
        if not self.is_ragged:
            return None
        t_local = 0 if not self._schema.has(Dim.BATCH) else 0
        # In ragged, samples are (T_i, C), time is axis 0
        return [s.shape[0] for s in self._samples]

    @property
    def shape(self) -> tuple[int, ...]:
        if self.is_ragged:
            n_samples = len(self._samples)
            n_ch = self._samples[0].shape[-1]
            return (n_samples, -1, n_ch)  # -1 signals variable TIME
        return self._data.shape

    @property
    def dims(self) -> tuple[Dim, ...]:
        return self._schema.dims

    @property
    def schema(self) -> DimSchema:
        return self._schema

    @property
    def channels(self) -> ChannelIndex | None:
        return self._channels

    @property
    def ch(self) -> _ChannelAccessor:
        return _ChannelAccessor(self)

    @property
    def y(self) -> np.ndarray | None:
        return self._y

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def n_samples(self) -> int:
        if self.is_ragged:
            return len(self._samples)
        if self._schema.has(Dim.BATCH):
            return self._data.shape[self._schema.axis(Dim.BATCH)]
        return 1

    @property
    def n_channels(self) -> int:
        ch_dim = self._find_channel_dim(self._schema)
        if ch_dim is not None:
            if self.is_ragged:
                return self._samples[0].shape[-1]
            return self._data.shape[self._schema.axis(ch_dim)]
        return 0

    @property
    def dtype(self) -> np.dtype:
        if self.is_ragged:
            return self._samples[0].dtype
        return self._data.dtype

    # ------------------------------------------------------------------
    # Axis resolution
    # ------------------------------------------------------------------

    def axis(self, dim: Dim) -> int:
        """Resolve a semantic Dim to its integer axis index."""
        return self._schema.axis(dim)

    def has_dim(self, dim: Dim) -> bool:
        return self._schema.has(dim)

    # ------------------------------------------------------------------
    # Constructors that preserve metadata (the primary output API)
    # ------------------------------------------------------------------

    def with_data(
        self,
        data: np.ndarray,
        dims: tuple[Dim, ...] | DimSchema | None = None,
        channels: ChannelIndex | list[str] | None = None,
        y: np.ndarray | None = ...,  # sentinel: ... means "keep current"
        metadata: dict | None = None,
    ) -> "TimeTensor":
        """Return a new dense TimeTensor with updated data, preserving metadata.

        This is the primary way transforms produce output.
        """
        new_dims = dims if dims is not None else self._schema.dims
        if isinstance(new_dims, DimSchema):
            new_dims = new_dims.dims

        if isinstance(channels, list):
            channels = ChannelIndex(channels)
        new_channels = channels if channels is not None else self._channels

        new_y = self._y if y is ... else y
        new_meta = metadata if metadata is not None else self._metadata

        return TimeTensor(
            data=data,
            dims=new_dims,
            channels=new_channels,
            y=new_y,
            metadata=new_meta,
        )

    def with_ragged(
        self,
        samples: list[np.ndarray],
        dims: tuple[Dim, ...] | None = None,
        channels: ChannelIndex | list[str] | None = None,
        y: np.ndarray | None = ...,
        metadata: dict | None = None,
    ) -> "TimeTensor":
        """Return a new ragged TimeTensor with updated samples."""
        new_dims = dims if dims is not None else self._schema.dims
        if isinstance(new_dims, DimSchema):
            new_dims = new_dims.dims
        schema = DimSchema(new_dims)

        if isinstance(channels, list):
            channels = ChannelIndex(channels)
        new_channels = channels if channels is not None else self._channels

        new_y = self._y if y is ... else y
        new_meta = metadata if metadata is not None else self._metadata

        return TimeTensor._from_ragged_internal(
            samples=samples,
            schema=schema,
            channels=new_channels,
            y=new_y,
            metadata=new_meta,
        )

    def with_y(self, y: np.ndarray | None) -> "TimeTensor":
        """Return a copy with different labels."""
        if self.is_ragged:
            return self.with_ragged(self._samples, y=y)
        return self.with_data(self._data, y=y)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Extract the raw ndarray (dense mode only)."""
        return self.data.copy()

    def to_dense(self, pad_value: float = 0.0) -> "TimeTensor":
        """Convert ragged → dense by padding to the longest sample."""
        if not self.is_ragged:
            return self
        max_len = max(s.shape[0] for s in self._samples)
        n_ch = self._samples[0].shape[-1]
        padded = np.full(
            (len(self._samples), max_len, n_ch), pad_value, dtype=self.dtype
        )
        lengths = []
        for i, s in enumerate(self._samples):
            padded[i, : s.shape[0], :] = s
            lengths.append(s.shape[0])

        meta = {**self._metadata, "_lengths": lengths}
        return TimeTensor(
            data=padded,
            dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
            channels=self._channels,
            y=self._y,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Indexing (batch slicing)
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> "TimeTensor":
        """Slice along the batch axis."""
        # Handle tuple keys like X[indices, ...] from sklearn's _safe_indexing
        if isinstance(key, tuple):
            key = key[0]
        if self.is_ragged:
            if isinstance(key, (int, np.integer)):
                # Single sample — return as a 1-sample ragged
                new_y = np.array([self._y[key]]) if self._y is not None else None
                return TimeTensor._from_ragged_internal(
                    samples=[self._samples[key]],
                    schema=self._schema,
                    channels=self._channels,
                    y=new_y,
                    metadata=self._metadata,
                )
            if isinstance(key, slice):
                new_samples = self._samples[key]
                new_y = self._y[key] if self._y is not None else None
                return TimeTensor._from_ragged_internal(
                    samples=new_samples,
                    schema=self._schema,
                    channels=self._channels,
                    y=new_y,
                    metadata=self._metadata,
                )
            if isinstance(key, (list, np.ndarray)):
                key = np.asarray(key)
                new_samples = [self._samples[i] for i in key]
                new_y = self._y[key] if self._y is not None else None
                return TimeTensor._from_ragged_internal(
                    samples=new_samples,
                    schema=self._schema,
                    channels=self._channels,
                    y=new_y,
                    metadata=self._metadata,
                )
        else:
            b_ax = self._schema.axis(Dim.BATCH)
            if isinstance(key, (int, np.integer)):
                new_data = np.expand_dims(np.take(self._data, key, axis=b_ax), axis=b_ax)
                new_y = None
                if self._y is not None:
                    new_y = np.array([self._y[key]])
                return self.with_data(new_data, y=new_y)
            elif isinstance(key, slice):
                slices = [slice(None)] * self._data.ndim
                slices[b_ax] = key
                new_data = self._data[tuple(slices)]
                new_y = self._y[key] if self._y is not None else None
                return self.with_data(new_data, y=new_y)
            elif isinstance(key, (list, np.ndarray)):
                key = np.asarray(key)
                new_data = np.take(self._data, key, axis=b_ax)
                new_y = self._y[key] if self._y is not None else None
                return self.with_data(new_data, y=new_y)
            else:
                raise TypeError(f"Unsupported index type: {type(key)}")

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        mode = "ragged" if self.is_ragged else "dense"
        shape_str = (
            f"({self.n_samples}, [variable], {self.n_channels})"
            if self.is_ragged
            else str(self.shape)
        )
        dims_str = ", ".join(d.name for d in self.dims)
        y_str = f", y={self._y.shape}" if self._y is not None else ""
        return (
            f"TimeTensor({mode}, shape={shape_str}, "
            f"dims=({dims_str}){y_str})"
        )
