"""
Shape transforms: flatten, reshape, PCA, transpose.

These transforms change the tensor's dimensionality structure while
preserving or reducing semantic information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import Scope, Transform


# ======================================================================
# Flatten
# ======================================================================

class Flatten(Transform):
    """Flatten specified dims into a single FEATURE axis.

    By default, flattens all non-BATCH dims.  After flattening, the
    output is ``(B, M)`` where M is the product of the flattened axes.

    Parameters
    ----------
    dims : tuple of Dim or None
        Which dims to flatten.  If None, flattens everything except BATCH.
    """

    scope = Scope.GLOBAL

    def __init__(self, dims: tuple[Dim, ...] | None = None) -> None:
        super().__init__()
        self.target_dims = dims

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Flatten":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        data = X.data

        if self.target_dims is not None:
            # Flatten only specified dims
            axes_to_flatten = [X.axis(d) for d in self.target_dims if X.has_dim(d)]
            if not axes_to_flatten:
                return X
            axes_to_flatten.sort()

            # Move them to the end and merge
            keep_axes = [i for i in range(data.ndim) if i not in axes_to_flatten]
            perm = keep_axes + axes_to_flatten
            transposed = np.transpose(data, perm)

            # Merge the last len(axes_to_flatten) axes
            n_flatten = len(axes_to_flatten)
            new_shape = transposed.shape[:-n_flatten] + (-1,)
            result = transposed.reshape(new_shape)

            keep_dims = [X.dims[i] for i in keep_axes]
            new_dims = tuple(keep_dims) + (Dim.FEATURE,)
        else:
            # Flatten all except BATCH
            if X.has_dim(Dim.BATCH):
                b_ax = X.axis(Dim.BATCH)
                B = data.shape[b_ax]
                # Move batch to front
                perm = [b_ax] + [i for i in range(data.ndim) if i != b_ax]
                transposed = np.transpose(data, perm)
                result = transposed.reshape(B, -1)
                new_dims = (Dim.BATCH, Dim.FEATURE)
            else:
                result = data.ravel()[np.newaxis, :]
                new_dims = (Dim.BATCH, Dim.FEATURE)

        M = result.shape[-1]
        feat_names = [f"f_{i}" for i in range(M)]

        return X.with_data(
            result,
            dims=new_dims,
            channels=ChannelIndex(feat_names),
        )


# ======================================================================
# Reshape
# ======================================================================

class Reshape(Transform):
    """Reshape the tensor to a new dim schema.

    Parameters
    ----------
    target_dims : tuple of Dim
        New semantic dims.
    target_shape : tuple of int
        New shape (must be compatible with data size, -1 for inferred axis).
    """

    scope = Scope.GLOBAL

    def __init__(
        self,
        target_dims: tuple[Dim, ...],
        target_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.target_dims_tuple = target_dims
        self.target_shape = target_shape

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Reshape":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        result = X.data.reshape(self.target_shape)
        return X.with_data(result, dims=self.target_dims_tuple)


# ======================================================================
# Transpose
# ======================================================================

class Transpose(Transform):
    """Reorder the axes of a TimeTensor by semantic dim names.

    Parameters
    ----------
    order : tuple of Dim
        Desired axis order.
    """

    scope = Scope.GLOBAL

    def __init__(self, order: tuple[Dim, ...]) -> None:
        super().__init__()
        self.order = order

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Transpose":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        perm = [X.axis(d) for d in self.order]
        result = np.transpose(X.data, perm)
        return X.with_data(result, dims=self.order)


# ======================================================================
# PCA
# ======================================================================

class PCA(Transform):
    """Principal Component Analysis — reduces the CHANNEL/FEATURE axis.

    Flattens non-batch dims if needed, fits PCA, projects.

    Parameters
    ----------
    n_components : int or float
        Number of components, or variance ratio to keep (0 < float < 1).
    whiten : bool
        If True, divide components by singular values for unit variance.
    """

    scope = Scope.GLOBAL

    def __init__(self, n_components: int | float = 0.95, whiten: bool = False) -> None:
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.n_components_: int | None = None
        self._input_flat_size: int | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "PCA":
        data = X.data
        # Flatten to (B, M)
        if X.has_dim(Dim.BATCH):
            b_ax = X.axis(Dim.BATCH)
            B = data.shape[b_ax]
            perm = [b_ax] + [i for i in range(data.ndim) if i != b_ax]
            flat = np.transpose(data, perm).reshape(B, -1)
        else:
            flat = data.ravel()[np.newaxis, :]

        self._input_flat_size = flat.shape[1]

        # Center
        self.mean_ = np.mean(flat, axis=0)
        centered = flat - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained_var = (S**2) / (flat.shape[0] - 1)
        total_var = np.sum(explained_var)

        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumsum = np.cumsum(explained_var) / total_var
            self.n_components_ = int(np.searchsorted(cumsum, self.n_components) + 1)
        else:
            self.n_components_ = int(self.n_components)

        self.n_components_ = min(self.n_components_, flat.shape[1], flat.shape[0])
        self.components_ = Vt[: self.n_components_]
        self.explained_variance_ = explained_var[: self.n_components_]
        self._singular_values = S[: self.n_components_]
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        data = X.data
        if X.has_dim(Dim.BATCH):
            b_ax = X.axis(Dim.BATCH)
            B = data.shape[b_ax]
            perm = [b_ax] + [i for i in range(data.ndim) if i != b_ax]
            flat = np.transpose(data, perm).reshape(B, -1)
        else:
            flat = data.ravel()[np.newaxis, :]

        centered = flat - self.mean_
        projected = centered @ self.components_.T  # (B, n_components)

        if self.whiten:
            projected /= np.sqrt(self.explained_variance_[np.newaxis, :] + 1e-10)

        feat_names = [f"pc_{i}" for i in range(self.n_components_)]
        return X.with_data(
            projected,
            dims=(Dim.BATCH, Dim.FEATURE),
            channels=ChannelIndex(feat_names),
        )

    def get_params(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "whiten": self.whiten}


# ======================================================================
# Squeeze / Unsqueeze
# ======================================================================

class Squeeze(Transform):
    """Remove a size-1 dimension by semantic name."""

    scope = Scope.GLOBAL

    def __init__(self, dim: Dim) -> None:
        super().__init__()
        self.dim = dim

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Squeeze":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        ax = X.axis(self.dim)
        if X.data.shape[ax] != 1:
            raise ValueError(
                f"Cannot squeeze {self.dim}: size is {X.data.shape[ax]}, not 1."
            )
        new_data = np.squeeze(X.data, axis=ax)
        new_dims = X.schema.remove(self.dim)
        return X.with_data(new_data, dims=new_dims.dims)


class Unsqueeze(Transform):
    """Add a size-1 dimension at a given semantic position."""

    scope = Scope.GLOBAL

    def __init__(self, dim: Dim, position: int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.position = position

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "Unsqueeze":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        pos = self.position if self.position >= 0 else X.data.ndim + 1 + self.position
        new_data = np.expand_dims(X.data, axis=pos)
        new_dims = X.schema.insert(pos, self.dim)
        return X.with_data(new_data, dims=new_dims.dims)
