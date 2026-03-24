"""
Semantic dimension system for TimeTensor.

Every axis in a TimeTensor has a semantic role (batch, time, channel, etc.).
Transforms reason about *which* axis to operate on by name, never by integer index.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Sequence


class Dim(Enum):
    """Semantic dimension roles for tensor axes."""

    BATCH = auto()
    TIME = auto()
    CHANNEL = auto()
    WINDOW = auto()  # from SlidingWindow: number of windows (K)
    FRAME = WINDOW  # alias for WINDOW in audio context (same underlying value)
    FREQUENCY = auto()  # from FFT/STFT/Mel
    FEATURE = auto()  # from feature extraction (flattened)

    @classmethod
    def register(cls, name: str) -> "Dim":
        """Register a custom dimension at runtime.

        Returns a new Dim member that behaves identically to built-in ones.
        Idempotent — re-registering the same name returns the existing member.
        """
        name = name.upper()
        if name in cls.__members__:
            return cls.__members__[name]
        obj = object.__new__(cls)
        obj._value_ = max(m.value for m in cls) + 1
        obj._name_ = name
        cls._member_map_[name] = obj
        cls._value2member_map_[obj.value] = obj
        return obj

    def __repr__(self) -> str:
        return f"Dim.{self.name}"


class DimSchema:
    """Validates and indexes a tuple of Dim roles for a tensor.

    Enforces that each role appears at most once (except for custom dims,
    which are explicitly opted-in by the user).
    """

    __slots__ = ("_dims", "_index")

    def __init__(self, dims: Sequence[Dim]) -> None:
        dims = tuple(dims)
        self._validate(dims)
        self._dims = dims
        self._index: dict[Dim, int] = {d: i for i, d in enumerate(dims)}

    @staticmethod
    def _validate(dims: tuple[Dim, ...]) -> None:
        seen: set[Dim] = set()
        for d in dims:
            if d in seen:
                raise ValueError(f"Duplicate dimension role {d!r}. Each semantic role may appear at most once.")
            seen.add(d)

    def axis(self, dim: Dim) -> int:
        """Resolve a semantic Dim to its integer axis index."""
        if dim not in self._index:
            raise KeyError(f"{dim!r} not present in schema {self._dims}. Available: {list(self._index.keys())}")
        return self._index[dim]

    def has(self, dim: Dim) -> bool:
        return dim in self._index

    def replace(self, old: Dim, new: Dim) -> "DimSchema":
        """Return a new schema with *old* replaced by *new*."""
        new_dims = tuple(new if d == old else d for d in self._dims)
        return DimSchema(new_dims)

    def insert(self, position: int, dim: Dim) -> "DimSchema":
        """Return a new schema with *dim* inserted at *position*."""
        dims_list = list(self._dims)
        dims_list.insert(position, dim)
        return DimSchema(tuple(dims_list))

    def remove(self, dim: Dim) -> "DimSchema":
        """Return a new schema with *dim* removed."""
        new_dims = tuple(d for d in self._dims if d != dim)
        if len(new_dims) == len(self._dims):
            raise KeyError(f"{dim!r} not in schema.")
        return DimSchema(new_dims)

    @property
    def dims(self) -> tuple[Dim, ...]:
        return self._dims

    @property
    def ndim(self) -> int:
        return len(self._dims)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DimSchema):
            return self._dims == other._dims
        if isinstance(other, tuple):
            return self._dims == other
        return NotImplemented

    def __contains__(self, dim: Dim) -> bool:
        return self.has(dim)

    def __len__(self) -> int:
        return len(self._dims)

    def __iter__(self):
        return iter(self._dims)

    def __repr__(self) -> str:
        inner = ", ".join(d.name for d in self._dims)
        return f"DimSchema({inner})"
