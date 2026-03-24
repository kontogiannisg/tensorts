"""
sklearn compatibility layer for tensorts.

SklearnWrapper bridges tensorts Pipelines with sklearn evaluation tools
(cross_val_score, GridSearchCV, cross_validate, etc.).

Usage
-----
    from tensorts import Pipeline, StandardScaler, Flatten, SklearnWrapper
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    preprocessing = Pipeline([
        ("scaler", StandardScaler()),
        ("flatten", Flatten()),
    ])
    wrapper = SklearnWrapper(preprocessing, RandomForestClassifier())
    scores = cross_val_score(wrapper, X, y, cv=5)

How indexing works
------------------
sklearn CV splitters call X[train_idx] and X[test_idx].  TimeTensor supports
list/array indexing on the batch axis out of the box, so no conversion needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .core.tensor import TimeTensor
    from .transforms.base import Pipeline

try:
    from sklearn.base import BaseEstimator as _BaseEstimator
except ImportError:
    _BaseEstimator = object  # sklearn not required at import time


class SklearnWrapper(_BaseEstimator):
    """Bridge between a tensorts Pipeline and sklearn evaluation tools.

    Wraps a tensorts preprocessing Pipeline and a sklearn-compatible estimator.
    The wrapper's fit/predict/score methods accept TimeTensor directly —
    sklearn CV splitters work because TimeTensor supports batch-axis indexing.

    Parameters
    ----------
    pipeline : tensorts Pipeline
        Preprocessing pipeline whose final output must be a dense TimeTensor.
        It is flattened to (n_samples, n_features) before the estimator sees it.
    estimator : sklearn estimator
        Any object with fit / predict (and optionally predict_proba / score).

    Examples
    --------
    # cross_val_score
    wrapper = SklearnWrapper(pipeline, RandomForestClassifier())
    scores = cross_val_score(wrapper, X, y, cv=StratifiedKFold(5))

    # GridSearchCV
    param_grid = {"pipeline__scaler__eps": [1e-6, 1e-8], "estimator__n_estimators": [50, 100]}
    gs = GridSearchCV(wrapper, param_grid, cv=5)
    gs.fit(X, y)
    """

    def __init__(self, pipeline: "Pipeline", estimator: Any) -> None:
        self.pipeline = pipeline
        self.estimator = estimator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_numpy(self, X: "TimeTensor") -> np.ndarray:
        """Transform X through the pipeline and flatten to 2D numpy array."""
        Xt = self.pipeline.transform(X)
        return Xt.data.reshape(Xt.n_samples, -1)

    def _labels(self, X: "TimeTensor", y) -> np.ndarray | None:
        """Resolve labels: prefer explicit y, fall back to X.y."""
        if y is not None:
            return np.asarray(y)
        if hasattr(X, "y") and X.y is not None:
            return X.y
        return None

    # ------------------------------------------------------------------
    # sklearn estimator interface
    # ------------------------------------------------------------------

    def fit(self, X: "TimeTensor", y=None) -> "SklearnWrapper":
        y = self._labels(X, y)
        Xt = self.pipeline.fit_transform(X, y)
        # A transform like SlidingWindow may update labels — prefer Xt.y
        if Xt.y is not None:
            y = Xt.y
        X_np = Xt.data.reshape(Xt.n_samples, -1)
        self.estimator.fit(X_np, y)
        return self

    def _aggregate_window_preds(self, preds: np.ndarray, Xt: "TimeTensor", n_original: int) -> np.ndarray:
        """Aggregate window-level predictions to sample level using mean.

        SlidingWindow stores per-sample window counts in Xt.metadata["_window_counts"].
        When present and the prediction count differs from n_original, we reduce
        by averaging all windows that belong to the same source sample.
        """
        window_counts = Xt.metadata.get("_window_counts")
        if window_counts is None or len(preds) == n_original:
            return preds
        result = np.empty(n_original, dtype=preds.dtype)
        offset = 0
        for i, k in enumerate(window_counts):
            result[i] = preds[offset : offset + k].mean() if k > 0 else 0.0
            offset += k
        return result

    def predict(self, X: "TimeTensor") -> np.ndarray:
        Xt = self.pipeline.transform(X)
        X_np = Xt.data.reshape(Xt.n_samples, -1)
        preds = self.estimator.predict(X_np)
        return self._aggregate_window_preds(preds, Xt, X.n_samples)

    def predict_proba(self, X: "TimeTensor") -> np.ndarray:
        Xt = self.pipeline.transform(X)
        X_np = Xt.data.reshape(Xt.n_samples, -1)
        proba = self.estimator.predict_proba(X_np)
        window_counts = Xt.metadata.get("_window_counts")
        if window_counts is None or proba.shape[0] == X.n_samples:
            return proba
        result = np.zeros((X.n_samples, proba.shape[1]))
        offset = 0
        for i, k in enumerate(window_counts):
            result[i] = proba[offset : offset + k].mean(axis=0) if k > 0 else 0.0
            offset += k
        return result

    def score(self, X: "TimeTensor", y=None) -> float:
        y_external = self._labels(X, y)
        Xt = self.pipeline.transform(X)
        X_np = Xt.data.reshape(Xt.n_samples, -1)
        # Use Xt.y when the pipeline changed the sample count (e.g. SlidingWindow).
        # The estimator was trained on the expanded labels, so scoring must use them too.
        y_score = Xt.y if Xt.y is not None and (y_external is None or len(Xt.y) != len(y_external)) else y_external
        return self.estimator.score(X_np, y_score)

    # ------------------------------------------------------------------
    # sklearn clone / GridSearchCV interface
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters for sklearn clone() and GridSearchCV."""
        params: dict[str, Any] = {
            "pipeline": self.pipeline,
            "estimator": self.estimator,
        }
        if deep:
            for k, v in self.pipeline.get_params(deep=True).items():
                params[f"pipeline__{k}"] = v
            if hasattr(self.estimator, "get_params"):
                for k, v in self.estimator.get_params(deep=True).items():
                    params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params) -> "SklearnWrapper":
        """Set parameters using nested 'pipeline__step__param' format."""
        for key, value in params.items():
            if key == "pipeline":
                self.pipeline = value
            elif key == "estimator":
                self.estimator = value
            elif key.startswith("pipeline__"):
                self.pipeline.set_params(**{key[len("pipeline__") :]: value})
            elif key.startswith("estimator__"):
                self.estimator.set_params(**{key[len("estimator__") :]: value})
            else:
                raise ValueError(
                    f"Invalid parameter {key!r} for SklearnWrapper. "
                    "Use 'pipeline__step__param' or 'estimator__param' format."
                )
        return self

    def __repr__(self) -> str:
        return f"SklearnWrapper(\n  pipeline={self.pipeline!r},\n  estimator={self.estimator!r}\n)"
