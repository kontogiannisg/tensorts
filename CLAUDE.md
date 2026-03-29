# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tensorts** is a channel-aware, shape-flexible tensor pipeline library for multivariate time series, built on NumPy. It provides composable transforms following sklearn's fit/transform protocol, with first-class support for both dense `(B, T, C)` and ragged (variable-length) inputs.

## Running Tests

Tests use a custom harness (not pytest). Run each suite directly:

```bash
python tests/test_tensorts.py                  # 62 tests — core + all transforms
python tests/test_feature_extractor.py         # 45 tests — multi-fn extraction + flatten naming + PER_SAMPLE + simplified API
python tests/test_function_transform.py        # 21 tests — stateful/stateless wrapping
python tests/test_sklearn_compat_problems.py   # 4 end-to-end sklearn integration problems
python tests/test_partial_fit.py               # 15 tests — incremental learning
python tests/test_feature_union.py            # 18 tests — parallel extraction + mixed scopes + 4D windowed
```

There are no single-test flags. To run one test, comment out others or extract the test function and call it directly.

Each `.py` test file has a corresponding `.ipynb` notebook in `examples/` that mirrors it step-by-step with intermediate output for debugging:
- `examples/01_Core_and_Transforms.ipynb` — mirrors `test_tensorts.py`
- `examples/02_FunctionTransform.ipynb` — mirrors `test_function_transform.py`
- `examples/03_FeatureExtractor.ipynb` — mirrors `test_feature_extractor.py`
- `examples/04_Sklearn_Integration.ipynb` — mirrors `test_sklearn_compat_problems.py`
- `examples/HAR_Classification_with_tensorts.ipynb` — end-to-end UCI HAR classification

## Architecture

### Core (`tensorts/core/`)

- **`dims.py`** — `Dim` enum (BATCH, TIME, CHANNEL, FEATURE, WINDOW, FREQUENCY, etc.) and `DimSchema` for validation. Custom dims via `Dim.register("NAME")`. Duplicates forbidden in schemas.
- **`channels.py`** — `ChannelIndex` with glob (`acc_*`), regex, and group-based resolution. Slices in sync when the tensor is sliced.
- **`tensor.py`** — `TimeTensor` — the central data structure. Holds dense `np.ndarray` or ragged `list[np.ndarray]`, plus dims, channels, y labels, and metadata. Key methods: `.with_data()` (dense), `.with_ragged()` (ragged) — both return new tensors without mutation.

### Transforms (`tensorts/transforms/`)

- **`base.py`** — `Transform` ABC (with `partial_fit`/`partial_fit_transform`), `Pipeline`, `ChannelTransform`, `ChannelSelect`, `FeatureUnion`, `Scope` enum (PER_CHANNEL, PER_SAMPLE, GLOBAL, ELEMENTWISE)
- **`preprocessing.py`** — StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer, Interpolate, LabelEncoder
- **`temporal.py`** — SlidingWindow (the critical ragged→dense boundary), Pad, Truncate, Resample
- **`spectral.py`** — PreEmphasis, Butterworth, STFT, MelSpectrogram, LogTransform
- **`functional.py`** — FunctionTransform, StatefulFunctionTransform, FeatureExtractor (single-fn and multi-fn modes)
- **`features.py`** — `make_statistical_extractor()` factory with built-in stats
- **`shape.py`** — Flatten, Reshape, Transpose, PCA, Squeeze, Unsqueeze

### sklearn Integration (`tensorts/compat.py`)

`SklearnWrapper` wraps a tensorts `Pipeline` + sklearn estimator. The pipeline handles TimeTensor transforms; the wrapper flattens output to 2D numpy for the estimator. Supports `cross_val_score`, `GridSearchCV`, nested `get_params`/`set_params`. Automatically decodes integer predictions back to original labels when `LabelEncoder` is in the pipeline, and aggregates window-level predictions back to sample level when `SlidingWindow(flatten_batch=True)` is used.

## Key Design Patterns

**Non-destructive updates:** All transforms produce new TimeTensors via `.with_data()` or `.with_ragged()`. Original tensors are never mutated.

**FunctionTransform vs FeatureExtractor:** FunctionTransform preserves dim schema (ndim stays the same). FeatureExtractor changes it — use it for operations that alter axis semantics (FFT, feature extraction, spectrograms). FunctionTransform should NOT be used when the semantic meaning of an axis changes.

**SlidingWindow and label expansion:** `SlidingWindow(flatten_batch=True)` expands B samples into B×K windows, stores `_window_counts` in metadata, and expands y labels via `label_strategy` ("repeat" or "last"). `SklearnWrapper` uses `_window_counts` to aggregate predictions back to sample level. Default (`flatten_batch=False`) produces `(B, K, W, C)` for both dense and ragged.

**Multi-fn FeatureExtractor:** Concatenates results along `Dim.FEATURE`, stores `_feature_names` in metadata. When followed by `Flatten`, descriptive cross-product names are produced (e.g., `mean__acc_x` instead of `f_0`).

**Ragged tensors:** Transforms declare `supports_ragged = True/False`. Access samples via `.samples[i]`, never `.data` (which raises RuntimeError on ragged). Dense indexing: `X.data[i]` or `X[i].data[0]` (the latter preserves batch dim).

**Scalers compute per-channel statistics:** StandardScaler/MinMaxScaler pool across batch and time (`reduce_axes = (0, 1)`), computing one mean/std per channel. Individual samples won't span [0,1] exactly with MinMaxScaler — only the most extreme sample touches 0 or 1.

**FeatureUnion — parallel extraction with mixed scopes:** `FeatureUnion` runs multiple transforms on the same input and concatenates their outputs along `Dim.FEATURE`. It automatically detects shared leading dims (e.g., `BATCH, WINDOW`) and preserves them, only flattening trailing feature/channel dims. This enables two key workflows:
- **3D `(B, T, C)` → `(B, K_total)`**: standard feature extraction for ML classifiers.
- **4D `(B, W, T, C)` → `(B, W, K_total)`**: rolling features after `SlidingWindow(flatten_batch=False)` for temporal DL (Conv1D/LSTM).

All sub-transforms must agree on leading dims — mixing windowed `(B, W, ...)` and non-windowed `(B, ...)` outputs raises `ValueError`. `PER_SAMPLE` extractors treat WINDOW as a leading axis: each function sees `(T, C)` per window independently. Feature names are descriptive (e.g., `mean__ax`, `cross_0`). Works with `SklearnWrapper`, `cross_val_score`, `partial_fit`, and `get_params`/`set_params`.

**Incremental learning (`partial_fit`):** All transforms support `partial_fit()`. The base class falls back to `fit()` (full refit). Transforms with true incremental algorithms (StandardScaler via Welford's, MinMaxScaler, MaxAbsScaler, LabelEncoder, Interpolate) override it. `supports_partial_fit` class attribute indicates whether a transform does true incremental learning vs refitting. Pipeline chains `partial_fit_transform()` through steps. SklearnWrapper delegates to the pipeline and estimator's `partial_fit`.

## Dependencies

Only NumPy is required. scipy, pandas, and sklearn are optional (needed for spectral transforms, IO, and SklearnWrapper respectively).
