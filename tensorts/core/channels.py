"""
Channel indexing system for TimeTensor.

Provides named access to the channel axis — select by name, glob, regex,
or predefined groups. When a TimeTensor is sliced, the ChannelIndex is
sliced in sync so downstream transforms always see correct names.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Sequence

import numpy as np


class ChannelIndex:
    """Named index for the channel axis of a TimeTensor.

    Supports:
    - Access by name:  ci['accel_x']           → integer index
    - Access by list:  ci['accel_x', 'gyro_y'] → index array
    - Access by glob:  ci['accel_*']            → index array
    - Access by regex: ci[re.compile(r'acc.+')] → index array
    - Access by group: ci.group('accel')        → index array
    """

    __slots__ = ("_names", "_name_to_idx", "_groups")

    def __init__(
        self,
        names: Sequence[str],
        groups: dict[str, list[str] | str] | None = None,
    ) -> None:
        self._names = tuple(names)
        self._name_to_idx: dict[str, int] = {n: i for i, n in enumerate(names)}
        if len(self._name_to_idx) != len(names):
            dupes = [n for n in names if names.count(n) > 1]  # type: ignore[arg-type]
            raise ValueError(f"Duplicate channel names: {set(dupes)}")

        self._groups: dict[str, list[str]] = {}
        if groups:
            for gname, pattern in groups.items():
                self.add_group(gname, pattern)

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def add_group(self, name: str, channels: list[str] | str) -> None:
        """Define a named group of channels.

        *channels* can be an explicit list of names or a glob pattern string.
        """
        if isinstance(channels, str):
            # Treat as glob pattern
            matched = self._glob(channels)
            if not matched:
                raise KeyError(f"Glob pattern {channels!r} matched no channels. Available: {self._names}")
            self._groups[name] = matched
        else:
            for ch in channels:
                if ch not in self._name_to_idx:
                    raise KeyError(f"Channel {ch!r} not found.")
            self._groups[name] = list(channels)

    def group(self, name: str) -> np.ndarray:
        """Return integer indices for a named group."""
        if name not in self._groups:
            raise KeyError(f"Group {name!r} not found. Available: {list(self._groups.keys())}")
        return np.array([self._name_to_idx[n] for n in self._groups[name]])

    def group_names(self, name: str) -> list[str]:
        """Return channel names belonging to a group."""
        if name not in self._groups:
            raise KeyError(f"Group {name!r} not found.")
        return list(self._groups[name])

    # ------------------------------------------------------------------
    # Resolution: names / globs / regex → integer indices
    # ------------------------------------------------------------------

    def _glob(self, pattern: str) -> list[str]:
        return [n for n in self._names if fnmatch.fnmatch(n, pattern)]

    def _resolve_key(self, key) -> np.ndarray:
        """Resolve a flexible key to an array of integer indices."""
        if isinstance(key, str):
            # Check if it's a glob pattern
            if any(c in key for c in "*?[]"):
                matched = self._glob(key)
                if not matched:
                    raise KeyError(f"Glob {key!r} matched no channels.")
                return np.array([self._name_to_idx[n] for n in matched])
            # Single name
            if key not in self._name_to_idx:
                raise KeyError(f"Channel {key!r} not found. Available: {self._names}")
            return np.array([self._name_to_idx[key]])

        if isinstance(key, re.Pattern):
            matched = [n for n in self._names if key.search(n)]
            if not matched:
                raise KeyError(f"Regex {key.pattern!r} matched no channels.")
            return np.array([self._name_to_idx[n] for n in matched])

        if isinstance(key, (list, tuple)):
            indices = []
            names_out = []
            for k in key:
                idx = self._resolve_key(k)
                indices.append(idx)
                names_out.extend(self._names[i] for i in idx)
            return np.concatenate(indices)

        if isinstance(key, int):
            if key < 0 or key >= len(self._names):
                raise IndexError(f"Channel index {key} out of range [0, {len(self._names)}).")
            return np.array([key])

        raise TypeError(f"Unsupported key type: {type(key)}")

    def resolve(self, key) -> np.ndarray:
        """Public API: resolve any supported key to integer indices."""
        return self._resolve_key(key)

    def resolve_names(self, key) -> list[str]:
        """Resolve a key to channel names."""
        indices = self._resolve_key(key)
        return [self._names[i] for i in indices]

    # ------------------------------------------------------------------
    # Slicing — produce a new ChannelIndex for a subset
    # ------------------------------------------------------------------

    def slice(self, indices: np.ndarray | list[int]) -> "ChannelIndex":
        """Return a new ChannelIndex containing only the given integer indices.

        Groups are preserved if all their members are in the subset.
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        new_names = [self._names[i] for i in indices]
        new_name_set = set(new_names)

        new_groups = {}
        for gname, members in self._groups.items():
            surviving = [m for m in members if m in new_name_set]
            if surviving:
                new_groups[gname] = surviving

        return ChannelIndex(new_names, groups=new_groups)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def groups(self) -> dict[str, list[str]]:
        return dict(self._groups)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_idx

    def __getitem__(self, key) -> np.ndarray:
        return self._resolve_key(key)

    def __repr__(self) -> str:
        if len(self._names) <= 6:
            inner = ", ".join(self._names)
        else:
            inner = ", ".join(self._names[:3]) + f", ... ({len(self._names)} total)"
        groups_str = ""
        if self._groups:
            groups_str = f", groups={list(self._groups.keys())}"
        return f"ChannelIndex([{inner}]{groups_str})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChannelIndex):
            return self._names == other._names
        return NotImplemented

    @classmethod
    def auto(cls, n_channels: int, prefix: str = "ch") -> "ChannelIndex":
        """Create a ChannelIndex with auto-generated names: ch_0, ch_1, ..."""
        names = [f"{prefix}_{i}" for i in range(n_channels)]
        return cls(names)

    def concat(self, other: "ChannelIndex") -> "ChannelIndex":
        """Concatenate two ChannelIndex objects. Raises on name collision."""
        overlap = set(self._names) & set(other._names)
        if overlap:
            raise ValueError(f"Channel name collision on concat: {overlap}")
        merged_groups = {**self._groups, **other._groups}
        return ChannelIndex(list(self._names) + list(other._names), groups=merged_groups)
