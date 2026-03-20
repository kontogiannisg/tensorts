# tensorts/core — Foundation Layer

The `core` module defines the semantic data model that everything else builds on. It has three tightly coupled components:

```
Dim          →  semantic axis roles (enum)
DimSchema    →  ordered tuple of Dims with index resolution
ChannelIndex →  named metadata for the CHANNEL axis
TimeTensor   →  main container — ndarray + DimSchema + ChannelIndex + y + metadata
```

---

## Dim — Semantic Axis Roles

```python
from tensorts import Dim

Dim.BATCH      # sample axis (N)
Dim.TIME       # time axis (T)
Dim.CHANNEL    # sensor / feature axis (C)
Dim.WINDOW     # sliding window axis (K)  — alias: Dim.FRAME
Dim.FREQUENCY  # frequency axis (F)       — output of STFT, mel, FFT
Dim.FEATURE    # extracted feature axis   — output of FeatureExtractor multi-fn
```

Every axis in a `TimeTensor` has exactly one `Dim` role. Transforms reason about *which* axis to operate on by name, never by integer position — so `FeatureExtractor(input_dim=Dim.TIME)` works regardless of where TIME sits in the tensor.

### Custom dims

Register project-specific roles at runtime:

```python
DEPTH = Dim.register("DEPTH")   # returns a Dim member
X = TimeTensor(data, dims=(Dim.BATCH, DEPTH, Dim.CHANNEL))
```

`register` is idempotent — calling it twice with the same name returns the existing member.

---

## DimSchema — Axis Order + Index Resolution

```python
from tensorts.core.dims import DimSchema

schema = DimSchema([Dim.BATCH, Dim.TIME, Dim.CHANNEL])
schema.axis(Dim.TIME)     # → 1
schema.has(Dim.FREQUENCY) # → False
Dim.TIME in schema        # → True
len(schema)               # → 3

# Non-destructive mutations (returns a new DimSchema each time)
schema.replace(Dim.TIME, Dim.WINDOW)   # swap one role
schema.insert(2, Dim.FREQUENCY)        # insert before axis 2
schema.remove(Dim.BATCH)               # drop an axis
```

`DimSchema` enforces that each role appears **at most once** — duplicate roles raise `ValueError` at construction time.

---

## ChannelIndex — Named Channel Metadata

```python
from tensorts.core.channels import ChannelIndex

ci = ChannelIndex(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])

# Single name → integer index
ci["acc_x"]                          # array([0])

# Glob pattern → index array
ci["acc_*"]                          # array([0, 1, 2])
ci["*_z"]                            # array([2, 5])

# List of names
ci[["acc_x", "gyro_z"]]             # array([0, 5])

# Regex
import re
ci[re.compile(r"gyro_.")]           # array([3, 4, 5])

# Named groups
ci.add_group("accel", "acc_*")      # define via glob
ci.add_group("gyro",  ["gyro_x", "gyro_y", "gyro_z"])  # explicit list
ci.group("accel")                   # → index array

# Slice to a subset (preserves partial groups)
ci.slice([0, 1, 2])                 # ChannelIndex(acc_x, acc_y, acc_z)

# Auto-generated names
ChannelIndex.auto(4)                # ch_0, ch_1, ch_2, ch_3
ChannelIndex.auto(3, prefix="mic") # mic_0, mic_1, mic_2

# Concatenate two indexes (raises on name collision)
ci_a.concat(ci_b)
```

When a `TimeTensor` is sliced, the `ChannelIndex` slices in sync — downstream transforms always see correct names.

### Groups at construction time

```python
ci = ChannelIndex(
    ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    groups={"accel": "acc_*", "gyro": "gyro_*"},
)
```

---

## TimeTensor — Central Data Container

`TimeTensor` wraps a NumPy array with:
- A `DimSchema` (which axis is TIME, CHANNEL, etc.)
- A `ChannelIndex` (names for the CHANNEL/FEATURE axis)
- An optional `y` label vector aligned with BATCH
- A `metadata` dict (sampling rate, units, provenance, ...)

### Construction — dense

```python
import numpy as np
from tensorts import TimeTensor, Dim

data = np.random.randn(32, 500, 6)   # 32 samples, 500 time steps, 6 channels

X = TimeTensor(
    data,
    dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
    channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    y=np.array([0]*16 + [1]*16),
    metadata={"sampling_rate": 100},
)
```

### Construction — ragged (variable-length samples)

```python
samples = [np.random.randn(100 + i*50, 3) for i in range(10)]  # different lengths

X = TimeTensor.from_ragged(
    samples,
    channels=["acc_x", "acc_y", "acc_z"],
    y=np.arange(10),
)

X.is_ragged   # True
X.shape       # (10, None, 3)  — None indicates variable time
```

### Key properties

```python
X.shape        # tuple — numpy shape for dense; (B, None, C) for ragged
X.ndim         # rank
X.schema       # DimSchema
X.channels     # ChannelIndex or None
X.y            # label array or None
X.metadata     # dict
X.is_ragged    # bool
X.data         # ndarray (dense only; raises for ragged)
```

### Axis resolution

```python
X.axis(Dim.TIME)        # integer axis index
X.has_dim(Dim.CHANNEL)  # True / False
```

### Non-destructive updates

The canonical way to produce a modified tensor — transforms use these internally:

```python
# Dense: replace data, keep dims/channels/y/metadata unless overridden
Xt = X.with_data(new_array)
Xt = X.with_data(new_array, dims=(Dim.BATCH, Dim.FEATURE, Dim.CHANNEL))
Xt = X.with_data(new_array, channels=new_channel_index)

# Ragged: replace sample list
Xt = X.with_ragged(new_samples)
Xt = X.with_ragged(new_samples, channels=new_ci)
```

Both methods propagate `y`, `metadata`, and other unchanged attributes automatically.

### Channel access shorthand

```python
X.ch["acc_*"]          # returns a new TimeTensor with matched channels only
X.ch["acc_x"]          # single channel (still TimeTensor, not squeezed)
X.ch.group("accel")    # by named group
```

### Label (y) patterns

Two patterns are supported — choose one per project:

**Pattern 1 — Native (y travels with X):**
```python
X = TimeTensor.from_ragged(samples, y=labels)
# Pass only X to pipeline; y is retrieved from Xt.y at the end
Xt = pipeline.fit_transform(X)
y_pred = model.predict(Xt.y)
```

**Pattern 2 — sklearn-compat (y separate):**
```python
X = TimeTensor.from_ragged(samples)   # no y
y = labels
# Pass both to sklearn APIs so they can slice X and y independently
cross_val_score(SklearnWrapper(pipeline, clf), X, y)
```

### IO utilities

```python
from tensorts import from_numpy, to_numpy, to_sklearn, to_dataframe

# Import
X = from_numpy(arr, sr=100, channel_names=["a", "b"])

# Export
arr = to_numpy(X)                      # ndarray
X_sk, y_sk = to_sklearn(X)            # (2D array, label array)
df = to_dataframe(X, sample_idx=0)    # pandas DataFrame for one sample
```

### Splitting

```python
from tensorts import kfold_temporal

for X_train, X_val in kfold_temporal(X, n_splits=5):
    ...
```
