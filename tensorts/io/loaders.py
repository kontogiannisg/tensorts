"""
Data loading utilities for creating TimeTensors from common formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim
from ..core.tensor import TimeTensor


def from_numpy(
    data: np.ndarray,
    dims: tuple[Dim, ...] = (Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    channels: list[str] | None = None,
    y: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> TimeTensor:
    """Create a TimeTensor from a NumPy array."""
    return TimeTensor(data=data, dims=dims, channels=channels, y=y, metadata=metadata)


def from_csv(
    path: str | Path,
    time_column: str | None = None,
    label_column: str | None = None,
    channel_columns: list[str] | None = None,
    group_by: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TimeTensor:
    """Load a TimeTensor from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    time_column : str, optional
        Column name for timestamps (excluded from channel data).
    label_column : str, optional
        Column name for labels. If *group_by* is set, one label per group.
    channel_columns : list of str, optional
        Which columns to use as channels.  If None, all numeric columns
        (excluding time_column and label_column) are used.
    group_by : str, optional
        Column name to group rows into separate samples (e.g., 'subject_id').
        Without this, the entire CSV is one sample.
    metadata : dict, optional
    """
    import csv

    path = Path(path)

    # Read CSV
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    # Determine channel columns
    exclude = set()
    if time_column:
        exclude.add(time_column)
    if label_column:
        exclude.add(label_column)
    if group_by:
        exclude.add(group_by)

    if channel_columns is None:
        channel_columns = []
        for col in fieldnames:
            if col in exclude:
                continue
            # Check if column is numeric
            try:
                float(rows[0][col])
                channel_columns.append(col)
            except (ValueError, TypeError):
                pass

    if not channel_columns:
        raise ValueError("No numeric channel columns found.")

    if group_by:
        # Group rows by the group_by column → ragged TimeTensor
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = row[group_by]
            groups.setdefault(key, []).append(row)

        samples = []
        labels = []
        for group_key, group_rows in groups.items():
            arr = np.array(
                [[float(r[c]) for c in channel_columns] for r in group_rows],
                dtype=np.float64,
            )
            samples.append(arr)
            if label_column:
                labels.append(group_rows[0][label_column])

        y = np.array(labels) if labels else None
        return TimeTensor.from_ragged(
            samples, channels=channel_columns, y=y, metadata=metadata
        )
    else:
        # Single sample or batch-is-rows
        data = np.array(
            [[float(r[c]) for c in channel_columns] for r in rows],
            dtype=np.float64,
        )
        y_arr = None
        if label_column:
            y_arr = np.array([r[label_column] for r in rows])

        # Shape: (T, C) → add batch dim → (1, T, C)
        data = data[np.newaxis, :, :]
        if y_arr is not None:
            y_arr = y_arr[np.newaxis]  # 1 sample

        return TimeTensor(
            data=data,
            dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
            channels=channel_columns,
            y=y_arr,
            metadata=metadata,
        )


def from_dataframe(
    df,
    time_column: str | None = None,
    label_column: str | None = None,
    channel_columns: list[str] | None = None,
    group_by: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TimeTensor:
    """Load a TimeTensor from a pandas DataFrame.

    Parameters are identical to ``from_csv``.
    """
    exclude = set()
    if time_column:
        exclude.add(time_column)
    if label_column:
        exclude.add(label_column)
    if group_by:
        exclude.add(group_by)

    if channel_columns is None:
        channel_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]

    if group_by:
        samples = []
        labels = []
        for key, group in df.groupby(group_by):
            arr = group[channel_columns].values.astype(np.float64)
            samples.append(arr)
            if label_column:
                labels.append(group[label_column].iloc[0])

        y = np.array(labels) if labels else None
        return TimeTensor.from_ragged(
            samples, channels=channel_columns, y=y, metadata=metadata
        )
    else:
        data = df[channel_columns].values.astype(np.float64)
        data = data[np.newaxis, :, :]
        y_arr = None
        if label_column:
            y_arr = np.array([df[label_column].iloc[0]])

        return TimeTensor(
            data=data,
            dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
            channels=channel_columns,
            y=y_arr,
            metadata=metadata,
        )
