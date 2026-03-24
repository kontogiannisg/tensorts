# tensorts

[![Tests](https://github.com/kontogiannisg/tensorts/actions/workflows/tests.yml/badge.svg)](https://github.com/kontogiannisg/tensorts/actions/workflows/tests.yml)
[![Ruff](https://github.com/kontogiannisg/tensorts/actions/workflows/lint.yml/badge.svg)](https://github.com/kontogiannisg/tensorts/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A channel-aware, shape-flexible tensor pipeline for multivariate time series.**

tensorts gives you a single container — `TimeTensor` — where every axis carries a semantic role (batch, time, channel, window, frequency, feature) and the channel axis carries named metadata. Transforms reason about *which axis to operate on by name*, never by integer index. The result is composable, domain-agnostic pipelines that work identically on `(B, T, C)` sensor data, `(B, FREQ, T', C)` spectrograms, and ragged lists of variable-length recordings.

---

## Why tensorts?

In practice, multivariate time series pipelines accumulate silent bugs:

- A scaler that was meant to work per-channel accidentally flattens all axes
- A spectrogram step inserts a new axis, shifting every integer index below it
- Variable-length recordings need padding before windowing, but the padding runs before scaling — leaking future statistics
- Labels go out of sync when windowing doubles the sample count

tensorts solves these by making the axis system explicit and carrying labels, channel names, and metadata through every transform automatically.

---

## Installation

```bash
pip install -e .               # core (numpy only)
pip install -e ".[all]"        # all optional deps (scipy, scikit-learn, pandas)
```

Optional dependency groups: `spectral` (scipy), `sklearn` (scikit-learn), `io` (pandas), `dev` (all + jupyter).

Run the test suite to verify:

```bash
python tests/test_tensorts.py                  # 62 tests
python tests/test_feature_extractor.py         # 31 tests
python tests/test_function_transform.py        # 21 tests
python tests/test_sklearn_compat_problems.py   # 4 real-world problems
```

---

## Package structure

```
tensorts/
├── __init__.py          — Public API (everything importable from top level)
│
├── core/                — Foundation layer
│   ├── dims.py          — Dim enum + DimSchema
│   ├── channels.py      — ChannelIndex: named channel access, groups, globs
│   └── tensor.py        — TimeTensor: the central data container
│
├── transforms/          — All transforms (sklearn fit/transform protocol)
│   ├── base.py          — Transform ABC, Pipeline, ChannelTransform, ChannelSelect, Scope
│   ├── preprocessing.py — StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
│   │                       Normalizer, PowerTransformer, QuantileTransformer,
│   │                       Interpolate, LabelEncoder
│   ├── temporal.py      — SlidingWindow, Pad, Truncate, Resample
│   ├── spectral.py      — PreEmphasis, ButterworthFilter, BandpassFilter, STFT,
│   │                       MelSpectrogram, LogTransform
│   ├── functional.py    — FunctionTransform, StatefulFunctionTransform, FeatureExtractor
│   ├── features.py      — make_statistical_extractor + built-in stat function map
│   └── shape.py         — Flatten, Reshape, Transpose, PCA, Squeeze, Unsqueeze
│
├── compat.py            — SklearnWrapper: cross_val_score / GridSearchCV bridge
│
├── io/                  — Data loading and export
│   ├── loaders.py       — from_numpy, from_csv, from_dataframe
│   └── exporters.py     — to_numpy, to_sklearn, to_dataframe
│
└── utils/               — Utilities
    └── split.py         — temporal_split, stratified_temporal_split, kfold_temporal
```

---

## Core concepts

### Dim — semantic axis roles

```python
from tensorts import Dim

Dim.BATCH      # samples / instances
Dim.TIME       # raw timesteps or within-window time
Dim.CHANNEL    # sensor channels, signal components
Dim.WINDOW     # window index from SlidingWindow (alias: Dim.FRAME)
Dim.FREQUENCY  # frequency bins from FFT / STFT / Mel
Dim.FEATURE    # extracted feature vector
```

Custom dims can be registered and are idempotent:

```python
scale_dim = Dim.register("SCALE")  # returns new Dim member
Dim.register("SCALE") is scale_dim  # True — safe to call repeatedly
```

### TimeTensor — the data container

A NumPy array where every axis has a semantic role. Operates in two modes:

| Mode | Storage | When |
|------|---------|------|
| **Dense** | Single `ndarray (B, T, C)` or higher rank | All samples same length |
| **Ragged** | List of `ndarray (T_i, C)` | Variable-length sequences |

```python
from tensorts import TimeTensor, Dim

# Dense — all samples 500 timesteps
ts = TimeTensor(
    data=np.random.randn(100, 500, 6),
    dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    channels=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
    channel_groups={'acc': 'acc_*', 'gyro': 'gyro_*'},
    y=labels,
    metadata={'sampling_rate': 50},
)

# Ragged — variable-length recordings
ts = TimeTensor.from_ragged(
    samples,                           # list of (T_i, C) arrays
    channels=['acc_x', 'acc_y', 'acc_z'],
    y=labels,
)

ts.shape           # (100, 500, 6) or (N, -1, C) for ragged
ts.axis(Dim.TIME)  # integer axis index, resolved by name
ts.n_samples       # 100
ts.n_channels      # 6
ts.y               # labels array or None
ts.metadata        # dict
ts.is_ragged       # bool

ts[0]              # single sample, preserves labels
ts[10:20]          # slice, preserves labels
ts[[0, 5, 9]]      # fancy index, preserves labels

ts.to_dense(pad_value=0.0)   # ragged → padded dense
ts.numpy()                   # raw ndarray (dense only)
```

### ChannelIndex — named channel access

```python
# Select by name, glob, or group
ts.ch['acc_x']         # single channel
ts.ch['acc_*']         # glob — returns acc_x, acc_y, acc_z
ts.ch.group('acc')     # predefined group

# Groups can be defined at creation or added later
ts.channels.add_group('accel_xy', ['acc_x', 'acc_y'])
```

### Y labels — two usage patterns

This is the most important design decision when using tensorts:

**Pattern 1: native tensorts** — embed `y` in the tensor, pass only `X`

```python
# y travels with X through every transform
X = TimeTensor.from_ragged(samples, y=labels)

pipe = Pipeline([...])
Xt = pipe.fit_transform(X)   # no y= argument needed
Xt.y                          # labels, updated by windowing if applicable

train, test = temporal_split(X, test_ratio=0.2)
pipe.fit(train)
Xt_test = pipe.transform(test)
Xt_test.y   # labels for the test set
```

**Pattern 2: sklearn compat** — pass `X` and `y` separately

```python
# Required for cross_val_score, GridSearchCV, etc.
# sklearn must slice y as a plain array independently of X.
scores = cross_val_score(wrapper, X, y=X.y, cv=5)
gs = GridSearchCV(wrapper, param_grid)
gs.fit(X, y=X.y)
```

The key difference: when `SlidingWindow(flatten_batch=True)` is in the pipeline, it expands B samples into B×K windows and propagates labels into `Xt.y`. In the sklearn path, `SklearnWrapper` uses `Xt.y` internally so the expanded label count matches the expanded sample count.

---

## Transforms

All transforms follow the sklearn fit/transform protocol and operate on `TimeTensor`. See [`tensorts/transforms/README.md`](tensorts/transforms/README.md) for the full reference.

### Scope — what each function sees

Transforms declare their operational scope:

| Scope | fn receives | Example use |
|-------|------------|-------------|
| `PER_CHANNEL` | 1-D vector `(T,)` per channel | scalers, filters |
| `PER_SAMPLE` | 2-D matrix `(T, C)` per sample | per-sample normalization |
| `GLOBAL` | full tensor | clipping, FFT over entire batch |

### Preprocessing

```python
from tensorts import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
    Interpolate, LabelEncoder,
)

StandardScaler()                          # zero-mean, unit-std per channel
MinMaxScaler(feature_range=(0, 1))        # scale to [0, 1] per channel
RobustScaler(quantile_range=(25, 75))     # median + IQR (outlier-robust)
MaxAbsScaler()                            # scale by max abs value → [-1, 1]
Normalizer(norm='l2')                     # per-sample L1/L2/max normalization
PowerTransformer(method='yeo-johnson')    # Yeo-Johnson → Gaussian per channel
QuantileTransformer(n_quantiles=1000)     # quantile mapping → uniform or normal
Interpolate(method='linear')              # fill NaN values
LabelEncoder()                            # encode string y labels to integers
```

All preprocessing transforms support ragged input and preserve it.

### Temporal

```python
from tensorts import SlidingWindow, Pad, Truncate, Resample

# Default: both dense and ragged → (B, K, W, C)
SlidingWindow(size=128, stride=64)

# flatten_batch=True: both → (B_total, W, C) with label expansion
SlidingWindow(size=128, stride=64, flatten_batch=True, label_strategy='repeat')

# Ragged → dense (pads to target_length)
Pad(target_length=256, mode='constant', value=0.0)

Truncate(target_length=200)   # trim time axis
Resample(target_length=64)    # linear interpolation to fixed length
```

`label_strategy` controls how labels expand with windows:
- `'repeat'` — all windows from sample i get label `y[i]`
- `'last'` — only the last window keeps the label

### Spectral

```python
from tensorts import PreEmphasis, ButterworthFilter, BandpassFilter, STFT, MelSpectrogram, LogTransform

PreEmphasis(coeff=0.97)                                   # high-freq boost; (B,T,C)→(B,T,C)
ButterworthFilter(cutoff=50, btype='lowpass', order=4)    # IIR filter; shape unchanged
BandpassFilter(low=1, high=20, sampling_rate=256)         # convenience bandpass
STFT(n_fft=512, hop_length=160)                           # (B,T,C)→(B,FREQ,T',C)
MelSpectrogram(n_fft=512, n_mels=64, sampling_rate=16000) # (B,T,C)→(B,mel,T',C)
LogTransform(eps=1e-10)                                   # log(x + eps), shape unchanged
```

### Shape

```python
from tensorts import Flatten, Reshape, Transpose, PCA, Squeeze, Unsqueeze

Flatten()                                         # (B,...) → (B, M)
Transpose(order=(Dim.BATCH, Dim.CHANNEL, Dim.TIME))
Squeeze(dim=Dim.WINDOW)                           # remove size-1 axis
Unsqueeze(dim=Dim.WINDOW, position=1)             # add size-1 axis
PCA(n_components=0.95)                            # keep 95% variance
PCA(n_components=32)                              # exact component count
```

### FunctionTransform — wrap any signal-processing function

`FunctionTransform` wraps a stateless function as a pipeline step. The dim schema is **preserved** — this is the right tool for preprocessing that changes values but not the semantic structure of the tensor.

```python
from tensorts import FunctionTransform, Scope
from scipy.signal import medfilt

# PER_CHANNEL: fn(T,) → (T',)  — applied independently to each channel
smooth = FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                            fn_kwargs={'kernel_size': 5})

# PER_SAMPLE: fn(T, C) → (T', C)  — applied independently to each sample
def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

normalize = FunctionTransform(fn=zscore, scope=Scope.PER_SAMPLE)

# GLOBAL: fn receives the full tensor
clip = FunctionTransform(fn=lambda x: np.clip(x, -3, 3), scope=Scope.GLOBAL)
```

**Important**: `FunctionTransform` preserves the number of dimensions (ndim). Use `FeatureExtractor` when a function changes the number of axes (e.g., `(T,) → (FREQ, T')` or `(T,) → scalar`).

### StatefulFunctionTransform — fit a state, then apply it

Like `FunctionTransform` but separates fit (learn statistics) from transform (apply them). The correct tool when you need to fit on training data and apply the learned state to test data.

```python
from tensorts import StatefulFunctionTransform, Scope

# PER_CHANNEL: learn a separate state per channel
adaptive_clip = StatefulFunctionTransform(
    fit_fn=lambda x: {'lo': np.percentile(x, 1), 'hi': np.percentile(x, 99)},
    transform_fn=lambda x, s: np.clip(x, s['lo'], s['hi']),
    scope=Scope.PER_CHANNEL,
)

# GLOBAL: one state for the whole tensor
scaler = StatefulFunctionTransform(
    fit_fn=lambda x: {'mu': x.mean(axis=(0,1), keepdims=True),
                      'sigma': x.std(axis=(0,1), keepdims=True)},
    transform_fn=lambda x, s: (x - s['mu']) / (s['sigma'] + 1e-8),
    scope=Scope.GLOBAL,
)
scaler.fit(train)
Xt_train = scaler.transform(train)
Xt_test  = scaler.transform(test)  # uses train statistics — no leakage
```

### FeatureExtractor — change dimensionality per channel

`FeatureExtractor` is for functions that **change the number of axes** for each channel. It operates in two modes:

#### Single-fn mode — one function, arbitrary output shape

```python
from tensorts import FeatureExtractor, Dim

# Spectrogram: (T,) → (FREQ, T')   TIME is replaced by FREQUENCY + TIME
fe = FeatureExtractor(
    fn=compute_mel,
    input_dim=Dim.TIME,
    output_dims=(Dim.FREQUENCY, Dim.TIME),
    fn_kwargs={'n_mels': 64},
)
# (B, T, C) → (B, FREQ, T', C)
# (B, W, T, C) → (B, W, FREQ, T', C)  — windowed input works automatically

# Scalar: (T,) → scalar   TIME axis removed
energy = FeatureExtractor(
    fn=lambda x: np.sum(x**2),
    input_dim=Dim.TIME,
    output_dims=(),     # empty tuple = scalar
)
# (B, T, C) → (B, C)
```

#### Multi-fn mode — multiple functions stacked along a FEATURE axis

This replaces the old `StatisticalFeatures` and `FrameFeatureBank`. Each function receives a 1-D vector and must return a scalar or 1-D array. Results are concatenated along a new `Dim.FEATURE` axis.

```python
# Statistical summary
fe = FeatureExtractor(
    functions=[
        ('mean',   np.mean),
        ('std',    np.std),
        ('energy', lambda x: float(np.sum(x**2))),
    ],
    input_dim=Dim.TIME,
)
# (B, T, C) → (B, 3, C)       feature_names_out_: ['mean', 'std', 'energy']
# (B, W, T, C) → (B, W, 3, C) — windowed input works automatically

# Mix scalar + vector outputs (K_total = sum of all output sizes)
fe = FeatureExtractor(
    functions=[
        ('zcr',    lambda x: float(np.mean(np.diff(np.signbit(x))))),  # scalar
        ('top5',   lambda x: np.sort(np.abs(np.fft.rfft(x)))[-5:]),    # 5-element
    ],
    input_dim=Dim.TIME,
)
# (B, T, C) → (B, 6, C)       feature_names_out_: ['zcr', 'top5__0', ..., 'top5__4']

# Per-function kwargs via 3-tuples
fe = FeatureExtractor(
    functions=[
        ('mean', np.mean),
        ('topk', compute_topk, {'k': 10}),   # (name, fn, kwargs_dict)
    ],
    input_dim=Dim.TIME,
)
```

### make_statistical_extractor — convenience factory

Pre-loads `FeatureExtractor` with common statistical functions. Accepts a mix of built-in names and custom callables:

```python
from tensorts import make_statistical_extractor

# Built-in names only (13 available)
fe = make_statistical_extractor(['mean', 'std', 'rms', 'skew', 'kurtosis', 'zcr'])

# Mix built-in + custom callable
def fft_rms(x):
    return float(np.sqrt(np.mean(np.abs(np.fft.rfft(x))**2)))

fe = make_statistical_extractor([
    'mean', 'std',
    ('fft_rms', fft_rms),                         # (name, callable)
    ('topk',    top_k_energy, {'k': 5}),           # (name, callable, kwargs)
])
```

Built-in names: `mean`, `std`, `var`, `min`, `max`, `median`, `rms`, `range`, `energy`, `skew`, `kurtosis`, `iqr`, `zcr`

---

## Pipeline composition

```python
from tensorts import Pipeline, ChannelTransform, ChannelSelect

# Sequential pipeline
pipe = Pipeline([
    ('scale',  StandardScaler()),
    ('window', SlidingWindow(size=128, stride=64)),
    ('fe',     make_statistical_extractor(['mean', 'std', 'rms'])),
    ('flat',   Flatten()),
])

Xt = pipe.fit_transform(train)
Xt_test = pipe.transform(test)

# Access steps by name
pipe['scale']   # the StandardScaler instance

# sklearn-style nested get_params / set_params
params = pipe.get_params()          # includes 'scale__eps', 'window__size', etc.
pipe.set_params(window__size=256)   # update nested parameter
```

### ChannelTransform — different transforms per channel group

```python
# Fit and apply different transforms to different channel groups
ct = ChannelTransform(
    transforms={
        'acc':  StandardScaler(),
        'gyro': RobustScaler(),
    },
    remainder='passthrough',  # or 'drop' to discard unmatched channels
)
```

### ChannelSelect — transform a subset, pass the rest through

```python
# Apply BandpassFilter only to EEG channels; others pass through unchanged
cs = ChannelSelect('eeg_*', BandpassFilter(low=0.5, high=40, sampling_rate=256))
```

---

## sklearn integration via SklearnWrapper

`SklearnWrapper` bridges a tensorts `Pipeline` with sklearn CV tools. It handles the `y` mismatch that arises when `SlidingWindow` expands samples into windows.

```python
from tensorts import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

preprocessing = Pipeline([
    ('scale',  StandardScaler()),
    ('window', SlidingWindow(size=128, stride=64, flatten_batch=True, label_strategy='repeat')),
    ('fe',     make_statistical_extractor(['mean', 'std', 'rms'])),
    ('flat',   Flatten()),
])

wrapper = SklearnWrapper(preprocessing, RandomForestClassifier(n_estimators=100))

# cross_val_score — pass X and y separately (sklearn slices both independently)
scores = cross_val_score(wrapper, X, y=X.y, cv=StratifiedKFold(5))

# GridSearchCV — nested parameter format
param_grid = {
    'pipeline__window__size': [64, 128, 256],
    'pipeline__window__stride': [32, 64],
    'estimator__n_estimators': [50, 100, 200],
}
gs = GridSearchCV(wrapper, param_grid, cv=5, scoring='accuracy')
gs.fit(X, y=X.y)
print(gs.best_params_)
```

**How it handles windowing**: when `SlidingWindow(flatten_batch=True)` expands B samples into B×K windows, `SklearnWrapper.fit()` uses `Xt.y` (the expanded labels) to fit the estimator. `predict()` averages window-level predictions back to sample level using `_window_counts` stored in metadata. `score()` uses `Xt.y` when sample counts differ.

---

## End-to-end examples

### HAR (Human Activity Recognition) — ragged recordings

```python
from tensorts import (
    TimeTensor, Pipeline, ChannelTransform,
    StandardScaler, RobustScaler, SlidingWindow, Flatten,
    temporal_split, make_statistical_extractor,
)

# Ragged: each recording has a different length
ts = TimeTensor.from_ragged(
    recordings,
    channels=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
    channel_groups={'acc': 'acc_*', 'gyro': 'gyro_*'},
    y=activity_labels,
    metadata={'sampling_rate': 50},
)

train, test = temporal_split(ts, test_ratio=0.2)

pipe = Pipeline([
    ('scale',  ChannelTransform({'acc': StandardScaler(), 'gyro': RobustScaler()})),
    ('window', SlidingWindow(size=128, stride=64, flatten_batch=True, label_strategy='repeat')),
    ('stats',  make_statistical_extractor(['mean', 'std', 'rms', 'skew', 'kurtosis'])),
    ('flat',   Flatten()),
])

X_train = pipe.fit_transform(train)   # y embedded in train, no separate y needed
X_test  = pipe.transform(test)

# 5 stats × 6 channels = 30 features per window
print(X_train.shape)   # (n_windows, 30)
print(X_train.y)       # window-level labels
```

### Audio — mel spectrogram pipeline

```python
from tensorts import (
    TimeTensor, Dim, Pipeline,
    PreEmphasis, MelSpectrogram, LogTransform, StandardScaler,
)

ts = TimeTensor(
    data=audio_data,    # (B, 16000, 1)
    dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    channels=['audio'],
    y=labels,
    metadata={'sampling_rate': 16000},
)

pipe = Pipeline([
    ('pe',    PreEmphasis(coeff=0.97)),
    ('mel',   MelSpectrogram(n_fft=512, hop_length=160, n_mels=64, sampling_rate=16000)),
    ('log',   LogTransform()),
    ('scale', StandardScaler()),
])
result = pipe.fit_transform(ts)
# (B, 16000, 1) → (B, 64, 97, 1)  — 64 mel bins × 97 frames
```

### Custom feature extraction with FeatureExtractor

```python
from tensorts import FeatureExtractor, Dim

# Single-fn: compute STFT per channel
stft_fe = FeatureExtractor(
    fn=lambda x: np.abs(np.fft.rfft(x, n=512)) ** 2,
    input_dim=Dim.TIME,
    output_dims=(Dim.FREQUENCY,),
    name='power_spectrum',
)
# (B, T, C) → (B, 257, C)

# Multi-fn: statistical + spectral features
fe = FeatureExtractor(
    functions=[
        ('mean',   np.mean),
        ('std',    np.std),
        ('energy', lambda x: float(np.sum(x**2))),
        ('zcr',    lambda x: float(np.mean(np.diff(np.signbit(x))))),
    ],
    input_dim=Dim.TIME,
)
result = fe.fit_transform(ts)
# (B, T, C) → (B, 4, C)   — 4 features per channel
print(fe.feature_names_out_)  # ['mean', 'std', 'energy', 'zcr']
```

### Custom preprocessing with FunctionTransform

```python
from tensorts import FunctionTransform, StatefulFunctionTransform, Scope
from scipy.signal import butter, sosfiltfilt

# Stateless: apply same filter to every sample/channel
def apply_filter(x, fs=256):
    sos = butter(4, [1, 40], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x)

prefilter = FunctionTransform(fn=apply_filter, scope=Scope.PER_CHANNEL,
                               fn_kwargs={'fs': 256})

# Stateful: learn per-channel clipping bounds from training data
clipper = StatefulFunctionTransform(
    fit_fn=lambda x: {'lo': np.percentile(x, 0.5), 'hi': np.percentile(x, 99.5)},
    transform_fn=lambda x, s: np.clip(x, s['lo'], s['hi']),
    scope=Scope.PER_CHANNEL,
)

pipe = Pipeline([('prefilter', prefilter), ('clip', clipper), ...])
pipe.fit(train)         # clipper learns bounds from train only
pipe.transform(test)    # applies train bounds to test — no data leakage
```

---

## Data I/O

```python
from tensorts import from_numpy, from_csv, from_dataframe
from tensorts import to_numpy, to_sklearn, to_dataframe

# Loading
ts = from_numpy(arr, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                channels=['a', 'b', 'c'], y=labels)

ts = from_csv('sensors.csv',
              group_by='subject_id',    # creates ragged TimeTensor
              label_column='activity',
              channel_columns=['ax', 'ay', 'az'])

ts = from_dataframe(df, group_by='trial', label_column='label')

# Exporting
X_np, y_np = to_sklearn(ts)         # flattened (B, M) + labels for any sklearn model
arr = to_numpy(ts)                   # raw ndarray
df  = to_dataframe(ts, sample_index=0)  # single sample as pandas DataFrame
```

---

## Splitting utilities

```python
from tensorts import temporal_split, stratified_temporal_split, kfold_temporal

# Temporal order is always preserved — no random shuffling
train, test = temporal_split(ts, test_ratio=0.2)
train, val, test = temporal_split(ts, test_ratio=0.15, val_ratio=0.15)

# Preserves class balance within temporal order
train, test = stratified_temporal_split(ts, test_ratio=0.2)

# Expanding-window k-fold — all train indices < all test indices (no future leakage)
for train_fold, test_fold in kfold_temporal(ts, n_folds=5):
    pipe.fit(train_fold)
    result = pipe.transform(test_fold)
```

---

## Available transforms — quick reference

### Preprocessing
| Transform | Input | Output | Notes |
|-----------|-------|--------|-------|
| `StandardScaler` | `(B, T, C)` | same | zero-mean, unit-std per channel |
| `MinMaxScaler` | `(B, T, C)` | same | scale to `feature_range` per channel |
| `RobustScaler` | `(B, T, C)` | same | median/IQR per channel |
| `MaxAbsScaler` | `(B, T, C)` | same | scale by max abs → [-1, 1] |
| `Normalizer` | `(B, T, C)` | same | per-sample L1/L2/max norm (stateless) |
| `PowerTransformer` | `(B, T, C)` | same | Yeo-Johnson → Gaussian per channel |
| `QuantileTransformer` | `(B, T, C)` | same | quantile mapping → uniform/normal |
| `Interpolate` | `(B, T, C)` with NaN | same | linear / nearest / forward-fill |
| `LabelEncoder` | any | same | encodes `X.y` string→int |

### Temporal
| Transform | Input | Output | Notes |
|-----------|-------|--------|-------|
| `SlidingWindow` | `(B, T, C)` | `(B, K, W, C)` | adds `WINDOW` axis |
| `SlidingWindow` | ragged `(T_i, C)` | `(B, K, W, C)` | ragged→dense (requires equal K) |
| `SlidingWindow(flatten_batch=True)` | any | `(B_total, W, C)` | batch×windows merged |
| `Pad` | ragged | dense | pads to `target_length` |
| `Truncate` | any | shorter | trims `TIME` axis |
| `Resample` | any | fixed-length | linear interpolation |

### Spectral
| Transform | Input | Output | Notes |
|-----------|-------|--------|-------|
| `PreEmphasis` | `(B, T, C)` | same | high-freq boost |
| `ButterworthFilter` | `(B, T, C)` | same | lowpass / highpass / bandpass |
| `BandpassFilter` | `(B, T, C)` | same | convenience bandpass wrapper |
| `STFT` | `(B, T, C)` | `(B, FREQ, T', C)` | power spectrum |
| `MelSpectrogram` | `(B, T, C)` | `(B, mel, T', C)` | Mel filterbank |
| `LogTransform` | any | same | `log(x + eps)` elementwise |

### Functional
| Transform | Role | Dim schema |
|-----------|------|-----------|
| `FunctionTransform` | Wrap stateless fn; preprocessing | **preserved** |
| `StatefulFunctionTransform` | Wrap stateful fn pair (fit/transform) | **preserved** |
| `FeatureExtractor` (single-fn) | One fn changes axes | `TIME` → `output_dims` |
| `FeatureExtractor` (multi-fn) | N fns stacked | `TIME` → `FEATURE` |
| `make_statistical_extractor()` | Factory for multi-fn stats | `TIME` → `FEATURE` |

### Shape
| Transform | Input | Output |
|-----------|-------|--------|
| `Flatten` | `(B, ...)` | `(B, M)` |
| `Reshape` | any | target dims |
| `Transpose` | any | reordered axes |
| `PCA` | any | `(B, n_components)` |
| `Squeeze` | has size-1 dim | rank reduced by 1 |
| `Unsqueeze` | any | rank increased by 1 |

### Composition
| Transform | Purpose |
|-----------|---------|
| `Pipeline` | Sequential chain with named steps |
| `ChannelTransform` | Different transforms per channel group |
| `ChannelSelect` | Transform a subset; passthrough others |
| `SklearnWrapper` | Bridge to sklearn CV tools |

---

## Dependencies

| Package | Required | Used for |
|---------|----------|----------|
| `numpy` | yes | all tensor operations |
| `scipy` | recommended | Butterworth filters, STFT windowing, interpolation |
| `pandas` | optional | `from_dataframe` / `to_dataframe` |
| `sklearn` | optional | `SklearnWrapper`, cross-validation |

---

## License

MIT
