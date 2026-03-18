# tensorts

**A channel-aware, shape-flexible tensor pipeline for multivariate time series.**

tensorts gives you a single data container — `TimeTensor` — where every axis has a semantic role (batch, time, channel, window, frequency, feature) and the channel axis carries named metadata. Transforms reason about *which* axis to operate on by name, never by integer index. The result is composable pipelines that work across domains: sensor data, audio, financial time series, and anything shaped like `(B, T, C)`.

---

## Key features

- **Semantic dimensions**: axes are `Dim.BATCH`, `Dim.TIME`, `Dim.CHANNEL`, etc. — transforms resolve them automatically regardless of tensor rank
- **Named channel indexing**: select channels by name, glob, regex, or predefined groups (`ts.ch['accel_*']`)
- **Per-channel routing**: apply different transforms to different channel subsets via `ChannelTransform` and `ChannelSelect`
- **Ragged support**: variable-length sequences are first-class citizens — windowing converts ragged → dense inside the pipeline
- **Label propagation**: `y` labels travel with the tensor through every transform, including shape-changing ones like windowing
- **sklearn-compatible**: every transform follows `fit` / `transform` / `fit_transform` — stateful transforms learn on train, apply on test
- **Batteries included**: scalers, filters, STFT, mel spectrograms, statistical features, framed audio features, PCA, and more
- **Pure NumPy**: zero deep-learning dependencies, minimal footprint (only `numpy` + optional `scipy`)

---

## Installation

```bash
# Clone or copy the tensorts/ directory into your project
pip install numpy scipy
```

Run the test suite to verify:

```bash
python test_tensorts.py
# Expected: 32/32 passed, 0/32 failed — ALL TESTS PASSED
```

---

## Quick start

```python
import numpy as np
from tensorts import (
    TimeTensor, Dim, Pipeline,
    StandardScaler, SlidingWindow, StatisticalFeatures, Flatten,
    temporal_split,
)

# Create a TimeTensor
data = np.random.randn(100, 500, 6)  # 100 samples, 500 timesteps, 6 channels
labels = np.random.randint(0, 3, size=100)

ts = TimeTensor(
    data=data,
    dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    channels=['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'],
    channel_groups={'acc': 'acc_*', 'gyr': 'gyr_*'},
    y=labels,
    metadata={'sampling_rate': 50},
)

# Split (preserves temporal order)
train, test = temporal_split(ts, test_ratio=0.2)

# Build pipeline
pipe = Pipeline([
    ('scale',   StandardScaler()),
    ('window',  SlidingWindow(size=128, stride=64)),
    ('stats',   StatisticalFeatures(['mean', 'std', 'rms', 'skew'])),
    ('flatten', Flatten()),
])

# Fit on train, transform both
X_train = pipe.fit_transform(train)
X_test  = pipe.transform(test)

# Feed to any sklearn model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train.numpy(), X_train.y)
clf.predict(X_test.numpy())
```

---

## Package structure

```
tensorts/
│
├── __init__.py                     # Public API — everything importable from top level
│
├── core/                           # Foundation layer
│   ├── dims.py                     # Dim enum + DimSchema
│   ├── channels.py                 # ChannelIndex — named channel access
│   └── tensor.py                   # TimeTensor — the central data container
│
├── transforms/                     # All transforms (sklearn fit/transform protocol)
│   ├── base.py                     # Transform, Pipeline, ChannelTransform,
│   │                               # ChannelSelect, FrameFeature, FrameFeatureBank
│   ├── preprocessing.py            # StandardScaler, MinMaxScaler, RobustScaler,
│   │                               # Interpolate, LabelEncoder
│   ├── temporal.py                 # SlidingWindow, Pad, Truncate, Resample
│   ├── spectral.py                 # PreEmphasis, ButterworthFilter, BandpassFilter,
│   │                               # STFT, MelSpectrogram, LogTransform
│   ├── features.py                 # StatisticalFeatures, RollingFeatures,
│   │                               # ZCR, RMS, SpectralEntropy, SpectralCentroid,
│   │                               # SpectralRolloff, SpectralFlatness, MFCC
│   └── shape.py                    # Flatten, Reshape, Transpose, PCA,
│                                   # Squeeze, Unsqueeze
│
├── io/                             # Data loading and export
│   ├── loaders.py                  # from_csv, from_numpy, from_dataframe
│   └── exporters.py                # to_numpy, to_sklearn, to_dataframe
│
└── utils/                          # Utilities
    └── split.py                    # temporal_split, stratified_temporal_split,
                                    # kfold_temporal
```

---

## Architecture

### Core layer

#### `Dim` — semantic axis roles

Every axis in a `TimeTensor` has a semantic role. Transforms use these roles to find the right axis automatically.

| Dim | Meaning | Typical position |
|-----|---------|-----------------|
| `BATCH` | Samples / instances | 0 |
| `TIME` | Temporal axis (raw timesteps or within-window time) | 1 |
| `CHANNEL` | Sensor channels, features, or signal components | last |
| `WINDOW` / `FRAME` | Window index from `SlidingWindow` (aliases for the same dim) | inserted before TIME |
| `FREQUENCY` | Frequency bins from FFT / STFT / Mel | replaces TIME |
| `FEATURE` | Extracted feature vector | replaces TIME+CHANNEL after feature extraction |

Custom dims can be registered at runtime:

```python
Dim.register("SCALE")  # returns a new Dim member
```

#### `DimSchema` — axis validation

Wraps a tuple of `Dim` values and enforces that each role appears at most once. Provides axis resolution (`schema.axis(Dim.TIME) → int`), replacement, insertion, and removal.

#### `ChannelIndex` — named channel access

Tracks channel names, supports glob matching (`'accel_*'`), regex, named groups, and slicing. When a `TimeTensor` is channel-sliced, the `ChannelIndex` is sliced in sync.

```python
ts.ch['accel_x', 'accel_y']   # select by name
ts.ch['accel_*']               # select by glob
ts.ch.group('accel')           # select by predefined group
```

#### `TimeTensor` — the central container

A NumPy array with semantic dims, named channels, optional labels, and metadata. Operates in two modes:

| Mode | Storage | When to use |
|------|---------|------------|
| **Dense** | Single `ndarray` of shape `(B, T, C)` or higher rank | All samples same length |
| **Ragged** | List of `ndarray` with shapes `(T_i, C)` | Variable-length sequences |

Key methods:

| Method | Purpose |
|--------|---------|
| `ts.axis(Dim.TIME)` | Resolve semantic dim → integer index |
| `ts.ch[...]` | Channel selection (returns new TimeTensor) |
| `ts.with_data(new_arr, ...)` | New TimeTensor with updated array, preserving metadata |
| `ts.with_ragged(samples, ...)` | New ragged TimeTensor with updated samples |
| `ts.to_dense(pad_value=0)` | Convert ragged → dense by padding |
| `ts.numpy()` | Extract raw ndarray |
| `ts[0]`, `ts[:10]` | Batch slicing (preserves labels) |

### Transform layer

Every transform follows the sklearn protocol:

```python
class Transform:
    def fit(self, X: TimeTensor, y=None) -> Transform: ...
    def transform(self, X: TimeTensor) -> TimeTensor: ...
    def fit_transform(self, X: TimeTensor, y=None) -> TimeTensor: ...
```

Transforms declare their operational scope:

| Scope | Meaning | Example |
|-------|---------|---------|
| `GLOBAL` | Needs all axes together | PCA, Flatten |
| `PER_CHANNEL` | Independent per channel, vectorized over batch+time | StandardScaler, ButterworthFilter |
| `PER_SAMPLE` | Independent per sample, vectorized over channels | SlidingWindow |
| `ELEMENTWISE` | Independent per element | LogTransform, PreEmphasis |

All scopes use vectorized NumPy — no Python loops over individual samples or channels.

### Shape transformation model

Transforms can change the tensor's dimensionality. The dim system tracks what happened at each step:

| Transform | Input dims | Output dims |
|-----------|-----------|-------------|
| `StandardScaler` | `(B, T, C)` | `(B, T, C)` — same |
| `SlidingWindow` | `(B, T, C)` | `(B, K, T, C)` — adds WINDOW |
| `SlidingWindow` (ragged) | `list of (T_i, C)` | `(B_total, T, C)` — ragged→dense |
| `StatisticalFeatures` | `(B, K, T, C)` | `(B, K, F)` — TIME+CHANNEL→FEATURE |
| `Flatten` | `(B, K, F)` | `(B, M)` — all→FEATURE |
| `STFT` | `(B, T, C)` | `(B, F, T', C)` — TIME→FREQUENCY+TIME |
| `MelSpectrogram` | `(B, T, C)` | `(B, mel, T', C)` — mel-scaled |
| `FrameFeatureBank` | `(B, K, T, C)` | `(B, K, F)` — TIME+CHANNEL→FEATURE |
| `PCA` | any | `(B, F)` — reduces to components |

---

## Pipeline examples

### 1. Air quality → regression

```python
pipe = Pipeline([
    ('impute',  Interpolate(method='linear')),
    ('scale',   StandardScaler()),
    ('window',  SlidingWindow(size=24, stride=1)),     # 24-hour windows
    ('stats',   StatisticalFeatures(['mean', 'std', 'min', 'max'])),
])
# (1, 8760, 5) → (1, 8737, 20)  — 20 features per window
```

### 2. HAR (variable-length) → classification

```python
pipe = Pipeline([
    ('scale',   ChannelTransform({
                    'acc': StandardScaler(),
                    'gyr': RobustScaler(),
                })),
    ('window',  SlidingWindow(size=128, stride=64, label_strategy='repeat')),
    ('stats',   StatisticalFeatures(['mean', 'std', 'rms', 'skew', 'kurtosis'])),
    ('flatten', Flatten()),
])
# 50 ragged samples → (206, 30)  — windowed, features extracted, labels propagated
```

### 3. Audio → mel spectrogram

```python
pipe = Pipeline([
    ('preemph', PreEmphasis(coeff=0.97)),
    ('mel',     MelSpectrogram(n_fft=512, hop_length=160, n_mels=64, sampling_rate=16000)),
    ('log',     LogTransform()),
    ('scale',   StandardScaler()),
])
# (8, 16000, 1) → (8, 64, 97, 1)  — ready for 2D-CNN
```

### 4. Audio → framed spectral features

```python
pipe = Pipeline([
    ('frame',    SlidingWindow(size=512, stride=256)),
    ('features', FrameFeatureBank([
                     ZCR(),
                     RMS(),
                     SpectralEntropy(),
                     SpectralCentroid(n_fft=512, sr=16000),
                 ])),
    ('scale',    StandardScaler()),
])
# (4, 16000, 1) → (4, 61, 4)  — 4 features per frame, channels named
```

---

## Channel routing

Apply different transforms to different channel subsets:

```python
# Different scalers per group
ChannelTransform({
    'accel': StandardScaler(),
    'gyro':  RobustScaler(),
})

# Transform specific channels, pass others through unchanged
ChannelSelect('accel_*', ButterworthFilter(cutoff=20, btype='lowpass'))
```

---

## Train / test workflow

Splitting is outside the pipeline; fitting is inside:

```python
from tensorts import temporal_split, stratified_temporal_split

# Temporal split (last 20% = test)
train, test = temporal_split(ts, test_ratio=0.2)

# Stratified split (preserves class balance)
train, test = stratified_temporal_split(ts, test_ratio=0.2)

# Three-way split
train, val, test = temporal_split(ts, test_ratio=0.15, val_ratio=0.15)

# Expanding-window k-fold (no future leakage)
folds = kfold_temporal(ts, n_folds=5)
for train_fold, test_fold in folds:
    pipe.fit(train_fold)
    result = pipe.transform(test_fold)
```

---

## Data I/O

```python
from tensorts import from_numpy, from_csv, from_dataframe
from tensorts import to_numpy, to_sklearn, to_dataframe

# Loading
ts = from_numpy(arr, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=[...])
ts = from_csv('data.csv', group_by='subject_id', label_column='activity')
ts = from_dataframe(df, channel_columns=['col_a', 'col_b'], label_column='y')

# Exporting
X, y = to_sklearn(ts)               # flattened (B, M) + labels for sklearn
arr  = to_numpy(ts)                  # raw ndarray
df   = to_dataframe(ts, sample_index=0)  # single sample as pandas DataFrame
```

---

## Available transforms

### Preprocessing
| Transform | Description |
|-----------|-------------|
| `StandardScaler` | Zero-mean, unit-variance per channel |
| `MinMaxScaler` | Scale to `[min, max]` range per channel |
| `RobustScaler` | Median/IQR scaling (outlier-robust) |
| `Interpolate` | Fill NaN values (linear, nearest, forward-fill, mean) |
| `LabelEncoder` | Encode string labels to integers |

### Temporal
| Transform | Description |
|-----------|-------------|
| `SlidingWindow` | Fixed-size windows with stride; handles ragged→dense + label propagation |
| `Pad` | Pad time axis to target length (constant, edge, reflect) |
| `Truncate` | Truncate time axis to target length |
| `Resample` | Resample to new time-axis length via interpolation |

### Spectral
| Transform | Description |
|-----------|-------------|
| `PreEmphasis` | High-frequency boost filter |
| `ButterworthFilter` | IIR filter (lowpass, highpass, bandpass, bandstop) |
| `BandpassFilter` | Convenience wrapper for bandpass Butterworth |
| `STFT` | Short-Time Fourier Transform |
| `MelSpectrogram` | Mel-scaled spectrogram |
| `LogTransform` | Element-wise log(x + eps) |

### Feature extraction
| Transform | Description |
|-----------|-------------|
| `StatisticalFeatures` | Per-window stats: mean, std, var, min, max, median, rms, skew, kurtosis, iqr, range, energy, zcr |
| `RollingFeatures` | Rolling-window statistics along time axis |
| `FrameFeatureBank` | Multiple per-frame extractors with shared FFT cache |
| `ZCR` | Zero-crossing rate (frame feature) |
| `RMS` | Root mean square energy (frame feature) |
| `SpectralEntropy` | Spectral entropy (frame feature) |
| `SpectralCentroid` | Spectral centroid (frame feature) |
| `SpectralRolloff` | Spectral rolloff frequency (frame feature) |
| `SpectralFlatness` | Spectral flatness / Wiener entropy (frame feature) |
| `MFCC` | Mel-frequency cepstral coefficients (frame feature) |

### Shape
| Transform | Description |
|-----------|-------------|
| `Flatten` | Merge dims into single FEATURE axis |
| `Reshape` | Reshape to new dim schema |
| `Transpose` | Reorder axes by semantic dim names |
| `PCA` | Principal component analysis |
| `Squeeze` | Remove a size-1 dimension |
| `Unsqueeze` | Add a size-1 dimension |

### Composition
| Transform | Description |
|-----------|-------------|
| `Pipeline` | Sequential transform chain with named steps |
| `ChannelTransform` | Different transforms per channel group |
| `ChannelSelect` | Transform a subset of channels, passthrough the rest |

---

## Dependencies

| Package | Required | Used for |
|---------|----------|----------|
| `numpy` | yes | Core tensor operations |
| `scipy` | recommended | Butterworth filters, STFT windowing, interpolation |
| `pandas` | optional | `from_dataframe` / `to_dataframe` |

---

## License

MIT
