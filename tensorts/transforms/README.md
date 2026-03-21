# tensorts/transforms — Transform Catalogue

All transforms follow the same contract:

```python
transform.fit(X, y=None)   → self          # learn parameters (no-op for stateless)
transform.transform(X)     → TimeTensor    # apply
transform.fit_transform(X) → TimeTensor    # both in one call
transform.get_params()     → dict          # sklearn GridSearchCV compatible
transform.set_params(**kw) → self          # sklearn GridSearchCV compatible
```

---

## Design rules

| Rule | Rationale |
|---|---|
| Transforms never mutate their input | Immutability prevents subtle pipeline bugs |
| Axis resolution is always semantic (`Dim.TIME`) | Integer indices break when axes are reordered |
| `y` propagates automatically via `with_data`/`with_ragged` | Labels travel with the tensor through the whole pipeline |
| Stateless transforms set `_is_fitted = True` in `fit` | Consistent `is_fitted` check for all transforms |
| `supports_ragged = True` opts in to ragged input | Transforms that can't handle ragged raise early with a helpful message |

---

## Scope enum

```python
from tensorts.transforms import Scope

Scope.GLOBAL       # whole tensor at once (e.g. PCA)
Scope.PER_CHANNEL  # independent per channel, vectorized over batch+time
Scope.PER_SAMPLE   # independent per sample, vectorized over channels
Scope.ELEMENTWISE  # independent per element (e.g. log, pre-emphasis)
```

`Scope` is a declaration for documentation and routing — it is not enforced by the base class.

---

## Pipeline — Sequential composition

```python
from tensorts.transforms import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("window", SlidingWindow(size=128, stride=64)),
    ("extractor", make_statistical_extractor(["mean", "std", "energy"])),
])

Xt = pipe.fit_transform(X)

# Access steps by name or index
pipe["scaler"]
pipe.named_steps["scaler"]

# GridSearchCV-compatible nested params
pipe.set_params(window__size=256)
pipe.get_params()   # includes 'window__size', 'window__stride', ...
```

`Pipeline.fit` calls `fit_transform` on all intermediate steps and `fit` on the last. Label updates (from windowing) propagate automatically between steps.

---

## ChannelTransform — Different transforms per channel group

Apply distinct transforms to different channels. The canonical way to handle heterogeneous sensor streams.

```python
from tensorts.transforms import ChannelTransform

ct = ChannelTransform(
    transforms={
        "acc_*":  StandardScaler(),      # glob
        "gyro":   MinMaxScaler(),         # named group
    },
    remainder="passthrough",   # or 'drop' to discard unmatched channels
)
```

- Keys can be channel names, glob patterns (`acc_*`), or group names defined on the `ChannelIndex`.
- `remainder="passthrough"` keeps unmatched channels unchanged.
- `remainder="drop"` discards them.
- Works on dense and ragged tensors.

---

## ChannelSelect — Apply one transform to a channel subset

```python
from tensorts.transforms import ChannelSelect

cs = ChannelSelect(
    channels="acc_*",          # glob, group name, or list of names
    transform=BandpassFilter(low_hz=0.5, high_hz=20, fs=100),
)
```

Selected channels are transformed; all others pass through unchanged. Output channel order: transformed first, then remainder.

---

## FunctionTransform — Stateless function wrapper

Wraps any stateless function that **preserves tensor rank** (ndim stays the same; shape may change along one axis).

```python
from tensorts.transforms import FunctionTransform, Scope

# PER_CHANNEL (default): fn receives (T,) → (T',) per channel
smooth = FunctionTransform(
    fn=lambda x: np.convolve(x, np.ones(5)/5, mode="same"),
    scope=Scope.PER_CHANNEL,
)

# PER_SAMPLE: fn receives (T, C) → (T', C) per sample
normalize = FunctionTransform(
    fn=lambda x: x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-8),
    scope=Scope.PER_SAMPLE,
)

# GLOBAL: fn receives the full ndarray → same-rank ndarray
invert = FunctionTransform(
    fn=lambda x: x[:, ::-1, :],   # reverse time
    scope=Scope.GLOBAL,
)
```

**`FunctionTransform` is NOT a feature extractor** — it never changes `ndim` or the dim schema. If your function changes the semantic meaning of an axis (e.g., FFT converts TIME → FREQUENCY), use `FeatureExtractor` with appropriate `output_dims` instead, so downstream transforms see the correct schema.

Supports ragged tensors.

---

## StatefulFunctionTransform — Stateful function wrapper

Like `FunctionTransform` but separates `fit` (learn from data) from `transform` (apply learned state).

```python
from tensorts.transforms import StatefulFunctionTransform, Scope

def fit_fn(samples: list, state: dict) -> None:
    """Called once on all data; write learned values into state."""
    all_data = np.concatenate(samples, axis=0)
    state["mean"] = np.mean(all_data, axis=0)
    state["std"] = np.std(all_data, axis=0) + 1e-8

def transform_fn(x: np.ndarray, state: dict) -> np.ndarray:
    """Called per sample; read from state."""
    return (x - state["mean"]) / state["std"]

scaler = StatefulFunctionTransform(
    fit_fn=fit_fn,
    transform_fn=transform_fn,
    scope=Scope.PER_SAMPLE,   # transform_fn receives (T, C) per sample
)
```

**Valid scope combinations:**

| `fit_scope` | `scope` | State container | Use case |
|---|---|---|---|
| `PER_CHANNEL` (default) | `PER_CHANNEL` | `_channel_states[i]` | Per-channel normalization |
| `PER_SAMPLE` | `PER_SAMPLE` | `_state` | Sample-level standardization |
| `PER_SAMPLE` | `GLOBAL` | `_state` | Fit per-sample, apply globally |
| `GLOBAL` | `GLOBAL` | `_state` | PCA-style global fit |

Do NOT mix `fit_scope=GLOBAL` with `scope=PER_CHANNEL` — they use different state containers and will error at transform time.

---

## FeatureExtractor — Dimensionality-changing feature extraction

Two modes, one class.

### Single-fn mode

One function that changes the dimensionality of the tensor. The consumed axis is replaced by `output_dims`.

```python
from tensorts.transforms import FeatureExtractor
from tensorts import Dim
import numpy as np

# Spectrogram: (T,) → (FREQ, TIME') per channel
spec = FeatureExtractor(
    fn=lambda x: np.abs(np.fft.rfft(x.reshape(-1, 64), axis=1)),
    input_dim=Dim.TIME,
    output_dims=(Dim.FREQUENCY, Dim.TIME),
)
# (B, T, C) → (B, FREQ, TIME', C)

# Scalar: (T,) → scalar — collapses time
energy = FeatureExtractor(
    fn=lambda x: float(np.sum(x**2)),
    input_dim=Dim.TIME,
    output_dims=(),   # scalar output → FEATURE axis with 1 element
)

# Mel spectrogram with kwargs
mel = FeatureExtractor(
    fn=compute_mel,
    fn_kwargs={"n_mels": 40, "sr": 16000},
    input_dim=Dim.TIME,
    output_dims=(Dim.FREQUENCY, Dim.TIME),
)
```

### Multi-fn mode (FeatureUnion semantics)

Multiple functions, each returning a scalar or 1-D array per channel. All results are concatenated along a new `Dim.FEATURE` axis.

```python
fe = FeatureExtractor(
    functions=[
        ("mean",   np.mean),                              # scalar
        ("std",    np.std),                               # scalar
        ("energy", lambda x: float(np.sum(x**2))),       # scalar
        ("mfcc",   compute_mfcc, {"n_mfcc": 13}),        # 1-D (13,) → 13 features
    ],
    input_dim=Dim.TIME,
)

# Dense: (B, T, C) → (B, K_total, C)  where K_total = 1+1+1+13 = 16
# Ragged: variable-length samples → dense (B, K_total, C)
```

**Output shapes:**

| Input | Output |
|---|---|
| `(B, T, C)` dense | `(B, K_total, C)` |
| ragged `(T_i, C)` samples | `(B, K_total, C)` dense |
| `(B, W, T, C)` windowed | `(B, W, K_total, C)` |

**Feature names** are populated in `fe.feature_names_out_` after first `transform`:
```
["mean__ch_0", "mean__ch_1", "std__ch_0", "std__ch_1", "mfcc__0__ch_0", ...]
```

Functions with kwargs are passed as 3-tuples `(name, fn, kwargs_dict)`:
```python
functions=[("top5", top_k_energy, {"k": 5})]
```

**`fn` and `functions` are mutually exclusive** — providing both raises `ValueError`.

---

## make_statistical_extractor — Convenience factory

Builds a multi-fn `FeatureExtractor` from a mix of built-in names and custom callables.

```python
from tensorts import make_statistical_extractor

# Built-ins only
fe = make_statistical_extractor(["mean", "std", "energy"])

# Mix built-ins with a custom function
def fft_rms(x):
    return float(np.sqrt(np.mean(np.abs(np.fft.rfft(x)) ** 2)))

fe = make_statistical_extractor(["mean", "std", ("fft_rms", fft_rms)])

# Custom function with kwargs (3-tuple)
def top_k_energy(x, k=5):
    return np.sort(np.abs(np.fft.rfft(x)))[-k:]

fe = make_statistical_extractor([
    "mean",
    ("top5_energy", top_k_energy, {"k": 5}),
])
```

Built-in names: `mean`, `std`, `var`, `min`, `max`, `median`, `rms`, `range`, `energy`, `skew`, `kurtosis`, `iqr`, `zcr`.

Default (no args): `["mean", "std", "min", "max"]`.

---

## Preprocessing

### StandardScaler

Zero-mean, unit-variance per channel. Computed over all batch and time samples pooled together.

```python
from tensorts.transforms import StandardScaler

scaler = StandardScaler(eps=1e-8)
scaler.fit(X_train)
Xt = scaler.transform(X_test)
Xt = scaler.inverse_transform(Xt)   # undo scaling

scaler.mean_   # learned per-channel mean
scaler.std_    # learned per-channel std
```

### MinMaxScaler

```python
from tensorts.transforms import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
```

### RobustScaler

Median + IQR based; robust to outliers.

```python
from tensorts.transforms import RobustScaler

scaler = RobustScaler()
```

### LabelEncoder

Encode string/categorical labels to integers and back.

```python
from tensorts.transforms import LabelEncoder

enc = LabelEncoder()
enc.fit(X)    # reads X.y
Xt = enc.transform(X)   # X.y becomes integer array
enc.inverse_transform(Xt)

enc.classes_   # original class names
```

### Interpolate

Fill NaN values via linear interpolation along the time axis.

```python
from tensorts.transforms import Interpolate

interp = Interpolate()   # stateless
Xt = interp.fit_transform(X)
```

---

## Temporal

### SlidingWindow

Extract fixed-size overlapping windows from the time axis.

```python
from tensorts.transforms import SlidingWindow

sw = SlidingWindow(
    size=128,               # window length in samples
    stride=64,              # step (default = size, no overlap)
    flatten_batch=False,    # default: 4D output (B, K, W, C)
    drop_last=True,         # discard partial last window
)
```

**Output shapes:**

| `flatten_batch` | Input | Output | Dims |
|---|---|---|---|
| `False` (default) | Dense `(B, T, C)` | `(B, K, W, C)` | `(BATCH, WINDOW, TIME, CHANNEL)` |
| `False` (default) | Ragged `[(T_i, C)]` | `(B, K, W, C)` | Same — requires all samples produce same K |
| `True` | Dense `(B, T, C)` | `(B*K, W, C)` | `(BATCH, TIME, CHANNEL)` |
| `True` | Ragged `[(T_i, C)]` | `(B_total, W, C)` | Same — K may differ per sample |

When `flatten_batch=True`, labels are expanded via `label_strategy` and `_window_counts` metadata is stored:
- `label_strategy="repeat"`: each window inherits its source sample's label
- `label_strategy="last"`: only the last window per sample is labelled (`-1` for others)

### Truncate

Clip or zero-pad to a fixed time length.

```python
from tensorts.transforms import Truncate

trunc = Truncate(length=256, pad=True)   # pad shorter sequences to length
```

### Resample

Resample along the time axis to a new number of samples or a target rate.

```python
from tensorts.transforms import Resample

rs = Resample(target_length=500)
# or
rs = Resample(target_sr=50, source_sr=100)   # downsample 2×
```

### Pad

Zero-pad (or other padding modes) along the time axis.

```python
from tensorts.transforms import Pad

pad = Pad(length=512, mode="zero")   # 'zero' | 'reflect' | 'replicate'
```

---

## Spectral

### PreEmphasis

High-frequency boost: `y[t] = x[t] - α·x[t-1]`. Standard first step in speech pipelines.

```python
from tensorts.transforms import PreEmphasis

pe = PreEmphasis(coeff=0.97)
```

### ButterworthFilter

IIR digital filter (lowpass, highpass, bandpass, bandstop).

```python
from tensorts.transforms import ButterworthFilter

lpf = ButterworthFilter(cutoff=20.0, order=4, btype="lowpass", fs=100)
bpf = ButterworthFilter(cutoff=(1.0, 40.0), order=2, btype="bandpass", fs=250)
```

### BandpassFilter

Convenience wrapper around `ButterworthFilter` for the common bandpass case.

```python
from tensorts.transforms import BandpassFilter

bp = BandpassFilter(low_hz=0.5, high_hz=20.0, fs=100, order=4)
```

### STFT

Short-time Fourier transform. Converts `(B, T, C) → (B, FREQ, TIME', C)`.

```python
from tensorts.transforms import STFT

stft = STFT(n_fft=512, hop_length=128, window="hann")
Xt = stft.fit_transform(X)   # Xt.schema → (BATCH, FREQUENCY, TIME, CHANNEL)
```

### MelSpectrogram

Mel-frequency spectrogram. Converts `(B, T, C) → (B, FREQ, TIME', C)`.

```python
from tensorts.transforms import MelSpectrogram

mel = MelSpectrogram(n_mels=40, n_fft=512, hop_length=128, sr=16000)
```

### LogTransform

Apply `log(1 + x)` element-wise. Typically used after spectral transforms to compress dynamic range.

```python
from tensorts.transforms import LogTransform

log = LogTransform()   # stateless
```

---

## Shape

### Flatten

Collapse multiple axes into a single `FEATURE` axis.

```python
from tensorts.transforms import Flatten
from tensorts import Dim

flat = Flatten()                            # flatten everything except BATCH
flat = Flatten(dims=(Dim.FREQUENCY, Dim.TIME))  # flatten only specific dims

# (B, FREQ, TIME', C) → (B, FREQ*TIME', C) or (B, FREQ*TIME'*C) depending on dims
```

**Descriptive feature names:** When Flatten follows a multi-fn `FeatureExtractor`, it produces cross-product names like `mean__acc_x`, `std__acc_y` instead of generic `f_0, f_1, ...`. This relies on `_feature_names` metadata set by FeatureExtractor.

### Reshape

Reshape to an explicit target shape (BATCH axis preserved).

```python
from tensorts.transforms import Reshape

r = Reshape(target_shape=(-1,))   # flatten per sample
```

### Transpose

Reorder axes by `Dim` role.

```python
from tensorts.transforms import Transpose

t = Transpose(dims=(Dim.BATCH, Dim.CHANNEL, Dim.TIME))
# (B, T, C) → (B, C, T)
```

### Squeeze / Unsqueeze

Add or remove size-1 axes by `Dim` role.

```python
from tensorts.transforms import Squeeze, Unsqueeze

sq = Squeeze(dim=Dim.FEATURE)      # remove size-1 FEATURE axis
us = Unsqueeze(dim=Dim.BATCH, position=0)  # insert BATCH axis
```

### PCA

Project to top-K principal components along the CHANNEL or FEATURE axis.

```python
from tensorts.transforms import PCA

pca = PCA(n_components=16)
Xt = pca.fit_transform(X)   # learns components on X; output has 16 channels

pca.components_          # (n_components, n_original_channels)
pca.explained_variance_ratio_
```

---

## sklearn compatibility

Any `Pipeline` (or individual transform) can be wrapped in `SklearnWrapper` to participate in `cross_val_score`, `GridSearchCV`, etc.:

```python
from tensorts import SklearnWrapper

wrapper = SklearnWrapper(pipeline, classifier)

# Pattern 1: y is embedded in X
cross_val_score(wrapper, X, cv=5)   # y=None; wrapper reads Xt.y

# Pattern 2: y passed separately (sklearn slices both X and y)
cross_val_score(wrapper, X, y, cv=5)

# GridSearchCV
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(wrapper, {"pipe__window__size": [64, 128, 256]}, cv=5)
gs.fit(X, y)
```

`SklearnWrapper` handles window-count mismatch in `predict`: when `SlidingWindow` expands one sample into K windows, `predict` aggregates K window predictions back to one sample prediction via majority vote.
