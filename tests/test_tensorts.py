"""
tensorts — comprehensive test suite.

Covers the full public API:
  1. Core: Dim, DimSchema, ChannelIndex, TimeTensor (dense + ragged)
  2. Y / label handling — native (embedded) vs sklearn-compat (separate) patterns
  3. IO utilities: from_numpy, to_numpy, to_sklearn, to_dataframe
  4. Preprocessing: StandardScaler, MinMaxScaler, RobustScaler, Interpolate, LabelEncoder
  5. Temporal: SlidingWindow, Pad, Truncate, Resample
  6. Shape: Flatten, Reshape, Transpose, Squeeze, Unsqueeze, PCA
  7. Spectral: PreEmphasis, ButterworthFilter, BandpassFilter, STFT, MelSpectrogram, LogTransform
  8. Transform base: Pipeline, ChannelTransform, ChannelSelect
  9. End-to-end pipelines (dense + ragged)
 10. Split utilities: temporal_split, stratified_temporal_split, kfold_temporal
"""

import traceback

import numpy as np

np.random.seed(42)

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception:
        FAIL += 1
        print(f"  ✗ {name}")
        traceback.print_exc()
        print()


# =====================================================================
# 1. CORE
# =====================================================================
print("=" * 60)
print("CORE TESTS")
print("=" * 60)


def test_dim_basics():
    from tensorts import Dim

    assert Dim.BATCH != Dim.TIME
    assert Dim.FRAME == Dim.WINDOW  # alias
    custom = Dim.register("MY_SCALE")
    assert custom.name == "MY_SCALE"
    assert Dim.register("MY_SCALE") is custom  # idempotent


test("Dim: basics, FRAME==WINDOW alias, custom registration", test_dim_basics)


def test_dim_schema():
    from tensorts import Dim, DimSchema

    s = DimSchema((Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    assert s.axis(Dim.TIME) == 1
    assert s.has(Dim.CHANNEL)
    assert not s.has(Dim.FREQUENCY)
    assert len(s) == 3

    try:
        DimSchema((Dim.BATCH, Dim.BATCH))
        assert False, "duplicate should raise"
    except ValueError:
        pass

    s2 = s.replace(Dim.TIME, Dim.FREQUENCY)
    assert s2.axis(Dim.FREQUENCY) == 1
    assert not s2.has(Dim.TIME)


test("DimSchema: axis resolution, duplicate detection, replace", test_dim_schema)


def test_channel_index():
    from tensorts import ChannelIndex

    ci = ChannelIndex(
        ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        groups={"acc": "acc_*", "gyro": "gyro_*"},
    )
    assert len(ci) == 6
    assert list(ci.group("acc")) == [0, 1, 2]
    assert list(ci.group("gyro")) == [3, 4, 5]
    assert ci.resolve_names("acc_*") == ["acc_x", "acc_y", "acc_z"]

    # Slicing preserves surviving groups
    sliced = ci.slice([0, 1, 2])
    assert "acc" in sliced.groups
    assert "gyro" not in sliced.groups
    assert len(sliced) == 3

    # Auto-naming
    auto = ChannelIndex.auto(4, prefix="ch")
    assert auto.names == ("ch_0", "ch_1", "ch_2", "ch_3")


test("ChannelIndex: names, globs, groups, slicing, auto-naming", test_channel_index)


def test_time_tensor_dense():
    from tensorts import Dim, TimeTensor

    data = np.random.randn(8, 100, 4)
    y = np.arange(8)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c", "d"],
        y=y,
        metadata={"sampling_rate": 100},
    )
    assert ts.shape == (8, 100, 4)
    assert ts.n_samples == 8
    assert ts.n_channels == 4
    assert not ts.is_ragged
    assert ts.y is not None
    assert ts.metadata["sampling_rate"] == 100
    assert ts.axis(Dim.TIME) == 1
    assert ts.has_dim(Dim.CHANNEL)

    # Channel accessor
    sub = ts.ch["a", "b"]
    assert sub.shape == (8, 100, 2)
    assert sub.channels.names == ("a", "b")

    # Glob accessor
    sub2 = ts.ch["a*"]
    assert sub2.n_channels == 1

    # Batch slicing: int → 1-sample
    s0 = ts[0]
    assert s0.n_samples == 1
    np.testing.assert_array_equal(s0.y, [0])

    # Batch slicing: slice
    mid = ts[2:5]
    assert mid.n_samples == 3
    np.testing.assert_array_equal(mid.y, [2, 3, 4])

    # Batch slicing: list
    sel = ts[[0, 3, 7]]
    assert sel.n_samples == 3


test("TimeTensor dense: shape, channels, y, metadata, batch slicing", test_time_tensor_dense)


def test_time_tensor_ragged():
    from tensorts import TimeTensor

    samples = [
        np.random.randn(100, 3),
        np.random.randn(200, 3),
        np.random.randn(150, 3),
    ]
    y = np.array([0, 1, 2])
    ts = TimeTensor.from_ragged(samples, channels=["x", "y", "z"], y=y)

    assert ts.is_ragged
    assert ts.n_samples == 3
    assert ts.lengths == [100, 200, 150]
    assert ts.shape == (3, -1, 3)

    # Channel selection on ragged preserves variability
    sub = ts.ch["x"]
    assert sub._samples[0].shape == (100, 1)
    assert sub._samples[1].shape == (200, 1)

    # Batch slicing preserves y
    s = ts[1:3]
    assert s.n_samples == 2
    np.testing.assert_array_equal(s.y, [1, 2])

    # to_dense pads to max length
    dense = ts.to_dense(pad_value=0.0)
    assert not dense.is_ragged
    assert dense.shape == (3, 200, 3)
    # Original data preserved, padding zeros
    np.testing.assert_array_equal(dense.data[0, :100, :], samples[0])
    np.testing.assert_array_equal(dense.data[0, 100:, :], 0.0)


test("TimeTensor ragged: from_ragged, lengths, channel access, slicing, to_dense", test_time_tensor_ragged)


def test_with_data_preserves_metadata():
    from tensorts import Dim, TimeTensor

    ts = TimeTensor(
        data=np.random.randn(4, 50, 2),
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b"],
        y=np.array([0, 1, 0, 1]),
        metadata={"sr": 100, "unit": "m/s"},
    )
    new = ts.with_data(np.random.randn(4, 50, 2))
    assert new.channels.names == ("a", "b")
    np.testing.assert_array_equal(new.y, ts.y)
    assert new.metadata == {"sr": 100, "unit": "m/s"}

    # with_y changes only labels
    relabeled = ts.with_y(np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(relabeled.y, [1, 0, 1, 0])
    assert relabeled.shape == ts.shape


test("with_data / with_y preserve metadata and labels", test_with_data_preserves_metadata)


# =====================================================================
# 2. Y / LABEL HANDLING — KEY PATTERNS
# =====================================================================
print()
print("=" * 60)
print("Y / LABEL HANDLING PATTERNS")
print("=" * 60)

# ── Pattern 1: Native tensorts usage ─────────────────────────────────
# y is embedded in the TimeTensor. Pass only X to the pipeline.
# This is the standard way to use tensorts outside sklearn CV.


def test_y_embedded_native_pattern():
    """y travels with X; pipeline sees it via X.y — no separate y argument needed."""
    from tensorts import Dim, Flatten, Pipeline, StandardScaler, TimeTensor

    data = np.random.randn(10, 50, 3)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # Embed labels at construction time
    X = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=y)

    # Pass only X — y is carried internally
    pipe = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    Xt = pipe.fit_transform(X)  # no y argument

    # Labels are still accessible on the output
    np.testing.assert_array_equal(Xt.y, y)

    # Train / test split — labels split with the data
    from tensorts import temporal_split

    train, test = temporal_split(X, test_ratio=0.2)
    np.testing.assert_array_equal(train.y, y[:8])
    np.testing.assert_array_equal(test.y, y[8:])

    pipe2 = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    Xt_train = pipe2.fit_transform(train)  # no y needed
    Xt_test = pipe2.transform(test)  # no y needed
    assert Xt_train.n_samples == 8
    assert Xt_test.n_samples == 2


test("Pattern 1 (native): y embedded in X, pass only X to pipeline", test_y_embedded_native_pattern)


# ── Pattern 2: sklearn CV usage ───────────────────────────────────────
# sklearn CV splitters call wrapper.fit(X_slice, y_slice).
# Pass X as TimeTensor AND y separately to cross_val_score / GridSearchCV.
# SklearnWrapper._labels() resolves which y to use internally.


def test_y_separate_sklearn_pattern():
    """sklearn CV needs y passed separately so it can slice it independently."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from tensorts import (
        Dim,
        Flatten,
        Pipeline,
        SklearnWrapper,
        StandardScaler,
        TimeTensor,
    )

    data = np.random.randn(30, 20, 3)
    y = np.random.randint(0, 3, 30)

    # y embedded in X (tensorts style)
    X = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=y)

    pipe = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    wrapper = SklearnWrapper(pipe, LogisticRegression(max_iter=200))

    # Pass X and y separately to sklearn — sklearn must slice both independently.
    # X.y is used as y here; sklearn slices it as a plain array.
    scores = cross_val_score(wrapper, X, y=X.y, cv=3)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)


test("Pattern 2 (sklearn): X + y passed separately to cross_val_score", test_y_separate_sklearn_pattern)


def test_y_propagation_through_sliding_window():
    """SlidingWindow expands sample count; y expands automatically to match windows."""
    from tensorts import SlidingWindow, TimeTensor

    samples = [np.random.randn(100, 2), np.random.randn(150, 2)]
    y = np.array([0, 1])
    X = TimeTensor.from_ragged(samples, y=y)

    win = SlidingWindow(size=50, stride=25, label_strategy="repeat", flatten_batch=True)
    Xt = win.fit_transform(X)

    # Each sample's windows all get the sample's label
    assert Xt.y is not None
    assert len(Xt.y) == Xt.n_samples
    # All windows from sample 0 → label 0
    counts = Xt.metadata.get("_window_counts")
    assert counts is not None
    n0 = counts[0]
    assert all(Xt.y[:n0] == 0)
    assert all(Xt.y[n0:] == 1)


test("y propagation: SlidingWindow expands labels with windows", test_y_propagation_through_sliding_window)


def test_label_encoder():
    """LabelEncoder encodes X.y (not the data) to integer class indices."""
    from tensorts import Dim, LabelEncoder, TimeTensor

    data = np.random.randn(9, 10, 2)
    y = np.array(["cat", "dog", "cat", "bird", "dog", "bird", "cat", "dog", "bird"])
    X = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=y)

    enc = LabelEncoder()
    Xt = enc.fit_transform(X)

    assert Xt.y is not None
    assert set(Xt.y).issubset({0, 1, 2})
    assert len(enc.classes_) == 3
    assert "bird" in enc.classes_


test("LabelEncoder: encodes string y labels to integers", test_label_encoder)


# =====================================================================
# 3. IO UTILITIES
# =====================================================================
print()
print("=" * 60)
print("IO UTILITIES")
print("=" * 60)


def test_from_numpy():
    from tensorts import Dim, from_numpy

    data = np.random.randn(5, 30, 3)
    ts = from_numpy(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"], y=np.arange(5))
    assert ts.shape == (5, 30, 3)
    assert ts.channels.names == ("a", "b", "c")
    assert ts.y is not None


test("from_numpy: creates TimeTensor from ndarray", test_from_numpy)


def test_to_numpy_and_to_sklearn():
    from tensorts import Dim, Flatten, TimeTensor, to_numpy, to_sklearn

    data = np.random.randn(4, 10, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.array([0, 1, 0, 1]))

    # to_numpy: raw array
    arr = to_numpy(ts)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, data)

    # to_sklearn: (n_samples, n_features) + y
    flat = Flatten().fit_transform(ts)
    X_np, y_np = to_sklearn(flat)
    assert X_np.shape == (4, 30)  # 10 * 3
    assert y_np.shape == (4,)


test("to_numpy, to_sklearn: export dense TimeTensor", test_to_numpy_and_to_sklearn)


def test_to_dataframe():
    from tensorts import Dim, TimeTensor, to_dataframe

    data = np.arange(12).reshape(1, 4, 3).astype(float)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["x", "y", "z"])
    df = to_dataframe(ts, sample_index=0)
    assert list(df.columns) == ["x", "y", "z"]
    assert df.shape == (4, 3)


test("to_dataframe: single sample → pandas DataFrame", test_to_dataframe)


# =====================================================================
# 4. PREPROCESSING TRANSFORMS
# =====================================================================
print()
print("=" * 60)
print("PREPROCESSING TRANSFORMS")
print("=" * 60)


def test_standard_scaler_dense():
    from tensorts import Dim, StandardScaler, TimeTensor

    data = np.random.randn(32, 100, 4) * 10 + 5
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = StandardScaler().fit_transform(ts)
    for c in range(4):
        assert abs(np.mean(result.data[:, :, c])) < 0.1
        assert abs(np.std(result.data[:, :, c]) - 1.0) < 0.1


test("StandardScaler: dense — zero mean, unit std", test_standard_scaler_dense)


def test_standard_scaler_ragged():
    from tensorts import StandardScaler, TimeTensor

    samples = [np.random.randn(np.random.randint(50, 150), 2) * 5 + 3 for _ in range(20)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    result = StandardScaler().fit_transform(ts)
    assert result.is_ragged
    all_data = np.concatenate(result._samples, axis=0)
    assert abs(np.mean(all_data[:, 0])) < 0.2


test("StandardScaler: ragged — preserves raggedness", test_standard_scaler_ragged)


def test_minmax_scaler():
    from tensorts import Dim, MinMaxScaler, TimeTensor

    data = np.random.randn(16, 50, 3) * 100
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)
    assert result.data.min() >= -0.01
    assert result.data.max() <= 1.01


test("MinMaxScaler: dense — scales to [0, 1]", test_minmax_scaler)


def test_robust_scaler():
    from tensorts import Dim, RobustScaler, TimeTensor

    data = np.random.randn(16, 50, 2) * 10
    data[0, 0, 0] = 1000  # outlier
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = RobustScaler().fit_transform(ts)
    # Median-based — outlier doesn't blow up the median
    assert abs(np.median(result.data[:, :, 0])) < 0.5


test("RobustScaler: dense — outlier-robust via median/IQR", test_robust_scaler)


def test_interpolate():
    from tensorts import Dim, Interpolate, TimeTensor

    data = np.array([[[1, 2], [np.nan, 4], [5, np.nan], [7, 8]]], dtype=float)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b"])
    result = Interpolate(method="linear").fit_transform(ts)
    assert not np.isnan(result.data).any()
    assert abs(result.data[0, 1, 0] - 3.0) < 0.1  # linearly interpolated
    assert abs(result.data[0, 2, 1] - 6.0) < 0.1


test("Interpolate: linear NaN filling", test_interpolate)


def test_maxabs_scaler_dense():
    from tensorts import Dim, MaxAbsScaler, TimeTensor

    data = np.random.randn(16, 50, 3) * 100
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    scaler = MaxAbsScaler()
    result = scaler.fit_transform(ts)
    assert result.data.min() >= -1.01
    assert result.data.max() <= 1.01
    # Round-trip
    restored = scaler.inverse_transform(result)
    np.testing.assert_allclose(restored.data, ts.data, atol=1e-6)


test("MaxAbsScaler: dense — scales to [-1, 1], round-trip", test_maxabs_scaler_dense)


def test_maxabs_scaler_ragged():
    from tensorts import MaxAbsScaler, TimeTensor

    samples = [np.random.randn(np.random.randint(50, 150), 2) * 50 for _ in range(20)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    scaler = MaxAbsScaler()
    result = scaler.fit_transform(ts)
    assert result.is_ragged
    all_data = np.concatenate(result._samples, axis=0)
    assert all_data.max() <= 1.01
    # Round-trip
    restored = scaler.inverse_transform(result)
    for i in range(len(samples)):
        np.testing.assert_allclose(restored._samples[i], ts._samples[i], atol=1e-6)


test("MaxAbsScaler: ragged — preserves raggedness, round-trip", test_maxabs_scaler_ragged)


def test_normalizer_l2():
    from tensorts import Dim, Normalizer, TimeTensor

    data = np.random.randn(8, 50, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = Normalizer(norm="l2").fit_transform(ts)
    assert result.shape == ts.shape
    # Each sample should have L2 norm ≈ 1
    for i in range(8):
        norm = np.sqrt(np.sum(result.data[i] ** 2))
        assert abs(norm - 1.0) < 0.01, f"Sample {i} norm={norm}"


test("Normalizer: L2 — per-sample unit norm", test_normalizer_l2)


def test_normalizer_l1():
    from tensorts import Dim, Normalizer, TimeTensor

    data = np.abs(np.random.randn(8, 50, 3)) + 0.1
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = Normalizer(norm="l1").fit_transform(ts)
    for i in range(8):
        norm = np.sum(np.abs(result.data[i]))
        assert abs(norm - 1.0) < 0.01, f"Sample {i} L1 norm={norm}"


test("Normalizer: L1 — per-sample unit L1 norm", test_normalizer_l1)


def test_normalizer_ragged():
    from tensorts import Normalizer, TimeTensor

    samples = [np.random.randn(80, 2), np.random.randn(120, 2)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    result = Normalizer(norm="l2").fit_transform(ts)
    assert result.is_ragged
    for s in result._samples:
        assert abs(np.sqrt(np.sum(s**2)) - 1.0) < 0.01


test("Normalizer: ragged — per-sample L2", test_normalizer_ragged)


def test_power_transformer_dense():
    from tensorts import Dim, PowerTransformer, TimeTensor

    # Skewed data
    data = np.exp(np.random.randn(50, 100, 2))
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    pt = PowerTransformer(standardize=True)
    result = pt.fit_transform(ts)
    assert result.shape == ts.shape
    assert pt.lambdas_ is not None
    assert len(pt.lambdas_) == 2
    # After transform + standardize: mean ≈ 0, std ≈ 1
    for c in range(2):
        ch_data = result.data[:, :, c].ravel()
        assert abs(np.mean(ch_data)) < 0.2, f"Channel {c} mean={np.mean(ch_data)}"
        assert abs(np.std(ch_data) - 1.0) < 0.2, f"Channel {c} std={np.std(ch_data)}"
    # Round-trip
    restored = pt.inverse_transform(result)
    np.testing.assert_allclose(restored.data, ts.data, atol=0.1)


test("PowerTransformer: dense — Yeo-Johnson + standardize, round-trip", test_power_transformer_dense)


def test_power_transformer_ragged():
    from tensorts import PowerTransformer, TimeTensor

    samples = [np.exp(np.random.randn(80, 2)) for _ in range(15)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    pt = PowerTransformer()
    result = pt.fit_transform(ts)
    assert result.is_ragged
    restored = pt.inverse_transform(result)
    for i in range(len(samples)):
        np.testing.assert_allclose(restored._samples[i], ts._samples[i], atol=0.1)


test("PowerTransformer: ragged — round-trip", test_power_transformer_ragged)


def test_quantile_transformer_uniform():
    from tensorts import Dim, QuantileTransformer, TimeTensor

    data = np.random.randn(50, 100, 3) * 10
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
    result = qt.fit_transform(ts)
    assert result.shape == ts.shape
    # Output should be roughly in [0, 1]
    assert result.data.min() >= -0.01
    assert result.data.max() <= 1.01
    # Round-trip
    restored = qt.inverse_transform(result)
    np.testing.assert_allclose(restored.data, ts.data, atol=0.5)


test("QuantileTransformer: uniform — maps to [0, 1], round-trip", test_quantile_transformer_uniform)


def test_quantile_transformer_ragged():
    from tensorts import QuantileTransformer, TimeTensor

    samples = [np.random.randn(80, 2) * 10 for _ in range(20)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    qt = QuantileTransformer(n_quantiles=50, output_distribution="uniform")
    result = qt.fit_transform(ts)
    assert result.is_ragged
    all_data = np.concatenate(result._samples, axis=0)
    assert all_data.min() >= -0.01
    assert all_data.max() <= 1.01


test("QuantileTransformer: ragged — maps to [0, 1]", test_quantile_transformer_ragged)


def test_minmax_inverse_transform():
    from tensorts import Dim, MinMaxScaler, TimeTensor

    data = np.random.randn(16, 50, 3) * 100
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    scaler = MinMaxScaler(feature_range=(0, 1))
    result = scaler.fit_transform(ts)
    restored = scaler.inverse_transform(result)
    np.testing.assert_allclose(restored.data, ts.data, atol=1e-6)


test("MinMaxScaler: inverse_transform round-trip", test_minmax_inverse_transform)


def test_robust_inverse_transform():
    from tensorts import Dim, RobustScaler, TimeTensor

    data = np.random.randn(16, 50, 2) * 10
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    scaler = RobustScaler()
    result = scaler.fit_transform(ts)
    restored = scaler.inverse_transform(result)
    np.testing.assert_allclose(restored.data, ts.data, atol=1e-6)


test("RobustScaler: inverse_transform round-trip", test_robust_inverse_transform)


def test_standard_inverse_ragged():
    from tensorts import StandardScaler, TimeTensor

    samples = [np.random.randn(80, 2) * 5 + 3 for _ in range(10)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    scaler = StandardScaler()
    result = scaler.fit_transform(ts)
    restored = scaler.inverse_transform(result)
    for i in range(len(samples)):
        np.testing.assert_allclose(restored._samples[i], ts._samples[i], atol=1e-6)


test("StandardScaler: ragged inverse_transform round-trip", test_standard_inverse_ragged)


# =====================================================================
# 5. TEMPORAL TRANSFORMS
# =====================================================================
print()
print("=" * 60)
print("TEMPORAL TRANSFORMS")
print("=" * 60)


def test_sliding_window_dense():
    from tensorts import Dim, SlidingWindow, TimeTensor

    data = np.arange(20, dtype=float).reshape(1, 20, 1)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = SlidingWindow(size=5, stride=5).fit_transform(ts)
    # 1 sample × 4 windows × 5 steps × 1 channel
    assert result.shape == (1, 4, 5, 1)
    assert result.has_dim(Dim.WINDOW)
    np.testing.assert_array_equal(result.data[0, 0, :, 0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(result.data[0, 1, :, 0], [5, 6, 7, 8, 9])


test("SlidingWindow: dense (B,T,C) → (B,K,W,C)", test_sliding_window_dense)


def test_sliding_window_ragged():
    from tensorts import SlidingWindow, TimeTensor

    # 100 samples → (100-50)/25+1 = 3 windows
    # 200 samples → (200-50)/25+1 = 7 windows
    samples = [np.random.randn(100, 3), np.random.randn(200, 3)]
    y = np.array([0, 1])
    ts = TimeTensor.from_ragged(samples, channels=["a", "b", "c"], y=y)

    result = SlidingWindow(size=50, stride=25, label_strategy="repeat", flatten_batch=True).fit_transform(ts)

    assert not result.is_ragged
    assert result.shape[0] == 10  # 3 + 7 windows total
    assert result.y is not None
    assert len(result.y) == 10
    assert all(result.y[:3] == 0)  # first 3 windows → label 0
    assert all(result.y[3:] == 1)


test("SlidingWindow: ragged → dense with label propagation (flatten_batch=True)", test_sliding_window_ragged)


def test_sliding_window_dense_flatten_batch():
    from tensorts import Dim, SlidingWindow, TimeTensor

    data = np.arange(40, dtype=float).reshape(2, 20, 1)
    y = np.array([0, 1])
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=y)
    result = SlidingWindow(size=5, stride=5, flatten_batch=True).fit_transform(ts)
    # 2 samples × 4 windows = 8 total, flattened
    assert result.shape == (8, 5, 1)
    assert result.dims == (Dim.BATCH, Dim.TIME, Dim.CHANNEL)
    assert result.metadata["_window_counts"] == [4, 4]
    assert len(result.y) == 8
    assert all(result.y[:4] == 0)
    assert all(result.y[4:] == 1)


test(
    "SlidingWindow: dense flatten_batch=True → (B*K,W,C) with label expansion", test_sliding_window_dense_flatten_batch
)


def test_sliding_window_ragged_4d():
    from tensorts import Dim, SlidingWindow, TimeTensor

    # Equal-length samples → same K → can stack to 4D
    samples = [np.random.randn(100, 3), np.random.randn(100, 3)]
    y = np.array([0, 1])
    ts = TimeTensor.from_ragged(samples, channels=["a", "b", "c"], y=y)
    result = SlidingWindow(size=50, stride=25).fit_transform(ts)  # default flatten_batch=False
    # (100-50)/25+1 = 3 windows each
    assert result.shape == (2, 3, 50, 3)
    assert result.has_dim(Dim.WINDOW)
    np.testing.assert_array_equal(result.y, y)  # labels unchanged (no expansion)


test("SlidingWindow: ragged (equal K) → 4D (B,K,W,C), default flatten_batch=False", test_sliding_window_ragged_4d)


def test_pad_ragged_to_dense():
    from tensorts import Pad, TimeTensor

    samples = [np.ones((80, 2)), np.ones((120, 2))]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])
    result = Pad(target_length=150, mode="constant", value=0.0).fit_transform(ts)
    assert not result.is_ragged
    assert result.shape == (2, 150, 2)
    np.testing.assert_array_equal(result.data[0, 80:, :], 0.0)


test("Pad: ragged → dense, zero-pads to target length", test_pad_ragged_to_dense)


def test_truncate():
    from tensorts import Dim, TimeTensor, Truncate

    data = np.random.randn(4, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = Truncate(target_length=60).fit_transform(ts)
    assert result.shape == (4, 60, 3)


test("Truncate: dense — trims time axis", test_truncate)


def test_resample():
    from tensorts import Resample, TimeTensor

    samples = [np.random.randn(100, 2), np.random.randn(200, 2), np.random.randn(150, 2)]
    ts = TimeTensor.from_ragged(samples, channels=["x", "y"])
    result = Resample(target_length=64).fit_transform(ts)
    # Ragged → all same length → dense
    assert not result.is_ragged
    assert result.shape == (3, 64, 2)


test("Resample: ragged → resampled dense (all same length)", test_resample)


# =====================================================================
# 6. SHAPE TRANSFORMS
# =====================================================================
print()
print("=" * 60)
print("SHAPE TRANSFORMS")
print("=" * 60)


def test_flatten():
    from tensorts import Dim, Flatten, TimeTensor

    data = np.random.randn(4, 10, 5, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"])
    result = Flatten().fit_transform(ts)
    assert result.shape == (4, 150)  # 10 * 5 * 3
    assert result.has_dim(Dim.FEATURE)


test("Flatten: (B,W,T,C) → (B,M)", test_flatten)


def test_transpose():
    from tensorts import Dim, TimeTensor, Transpose

    data = np.random.randn(4, 8, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = Transpose(order=(Dim.BATCH, Dim.CHANNEL, Dim.TIME)).fit_transform(ts)
    assert result.shape == (4, 3, 8)
    assert result.dims == (Dim.BATCH, Dim.CHANNEL, Dim.TIME)


test("Transpose: reorder axes by semantic dim", test_transpose)


def test_squeeze_unsqueeze():
    from tensorts import Dim, Squeeze, TimeTensor, Unsqueeze

    data = np.random.randn(4, 1, 10, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL))

    squeezed = Squeeze(dim=Dim.WINDOW).fit_transform(ts)
    assert squeezed.shape == (4, 10, 3)
    assert not squeezed.has_dim(Dim.WINDOW)

    unsqueezed = Unsqueeze(dim=Dim.WINDOW, position=1).fit_transform(squeezed)
    assert unsqueezed.shape == (4, 1, 10, 3)
    assert unsqueezed.has_dim(Dim.WINDOW)


test("Squeeze / Unsqueeze: remove and add size-1 axes", test_squeeze_unsqueeze)


def test_pca_n_components():
    from tensorts import PCA, Dim, TimeTensor

    rng = np.random.RandomState(0)
    latent = rng.randn(60, 2)
    mixing = rng.randn(2, 8)
    data = (latent @ mixing + rng.randn(60, 8) * 0.01)[:, np.newaxis, :]
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    pca = PCA(n_components=2)
    result = pca.fit_transform(ts)
    assert result.shape == (60, 2)
    assert result.channels.names[0] == "pc_0"


test("PCA: reduce to 2 components", test_pca_n_components)


def test_pca_variance_threshold():
    from tensorts import PCA, Dim, TimeTensor

    rng = np.random.RandomState(1)
    latent = rng.randn(100, 3)
    mixing = rng.randn(3, 10)
    data = (latent @ mixing + rng.randn(100, 10) * 0.01)[:, np.newaxis, :]
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    pca = PCA(n_components=0.99)  # keep 99% variance
    result = pca.fit_transform(ts)
    assert result.n_channels <= 10
    assert result.n_channels >= 1


test("PCA: variance threshold (n_components=0.99)", test_pca_variance_threshold)


# =====================================================================
# 7. SPECTRAL TRANSFORMS
# =====================================================================
print()
print("=" * 60)
print("SPECTRAL TRANSFORMS")
print("=" * 60)


def test_pre_emphasis():
    from tensorts import Dim, PreEmphasis, TimeTensor

    data = np.sin(np.linspace(0, 4 * np.pi, 200)).reshape(1, 200, 1)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = PreEmphasis(coeff=0.97).fit_transform(ts)
    assert result.shape == (1, 200, 1)
    # Ragged input
    samples = [np.random.randn(100, 2), np.random.randn(150, 2)]
    ts_rag = TimeTensor.from_ragged(samples)
    result_rag = PreEmphasis().fit_transform(ts_rag)
    assert result_rag.is_ragged


test("PreEmphasis: dense + ragged (coeff=0.97)", test_pre_emphasis)


def test_butterworth_filter():
    from tensorts import ButterworthFilter, Dim, TimeTensor

    sr = 1000
    t = np.linspace(0, 1, sr)
    # 10 Hz signal + 200 Hz noise
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
    data = signal[np.newaxis, :, np.newaxis]
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), metadata={"sampling_rate": sr})

    lpf = ButterworthFilter(cutoff=50, btype="lowpass", sampling_rate=sr)
    result = lpf.fit_transform(ts)
    assert result.shape == (1, sr, 1)
    # High-frequency component should be attenuated
    fft_out = np.abs(np.fft.rfft(result.data[0, :, 0]))
    fft_in = np.abs(np.fft.rfft(data[0, :, 0]))
    freqs = np.fft.rfftfreq(sr, d=1 / sr)
    idx_200 = np.argmin(np.abs(freqs - 200))
    assert fft_out[idx_200] < fft_in[idx_200] * 0.1


test("ButterworthFilter: lowpass attenuates high-frequency content", test_butterworth_filter)


def test_bandpass_filter():
    from tensorts import BandpassFilter, Dim, TimeTensor

    sr = 500
    data = np.random.randn(2, sr, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), metadata={"sampling_rate": sr})
    result = BandpassFilter(low=1, high=20, sampling_rate=sr).fit_transform(ts)
    assert result.shape == (2, sr, 3)


test("BandpassFilter: passes, shape preserved", test_bandpass_filter)


def test_stft():
    from tensorts import STFT, Dim, TimeTensor

    sr = 8000
    data = np.random.randn(2, sr, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = STFT(n_fft=256, hop_length=128).fit_transform(ts)
    # (B, FREQ, TIME', C)
    assert result.has_dim(Dim.FREQUENCY)
    assert result.has_dim(Dim.TIME)
    assert result.n_channels == 2
    assert result.shape[1] == 256 // 2 + 1  # 129 bins


test("STFT: (B,T,C) → (B,FREQ,T',C)", test_stft)


def test_mel_spectrogram():
    from tensorts import Dim, MelSpectrogram, TimeTensor

    sr = 16000
    data = np.random.randn(4, sr, 1)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), metadata={"sampling_rate": sr})
    result = MelSpectrogram(n_fft=512, hop_length=160, n_mels=64, sampling_rate=sr).fit_transform(ts)
    assert result.shape[0] == 4
    assert result.shape[1] == 64  # mel bins
    assert result.has_dim(Dim.FREQUENCY)


test("MelSpectrogram: (B,T,C) → (B,64_mels,T',C)", test_mel_spectrogram)


def test_log_transform():
    from tensorts import Dim, LogTransform, TimeTensor

    data = np.abs(np.random.randn(4, 50, 3)) + 0.01  # positive values
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = LogTransform(eps=1e-10).fit_transform(ts)
    assert result.shape == (4, 50, 3)
    # log maps to negatives for values < 1
    assert result.data.min() < data.min()


test("LogTransform: element-wise log, shape preserved", test_log_transform)


# =====================================================================
# 8. TRANSFORM BASE CLASSES
# =====================================================================
print()
print("=" * 60)
print("TRANSFORM BASE CLASSES")
print("=" * 60)


def test_pipeline_fit_transform():
    from tensorts import Dim, Flatten, Pipeline, StandardScaler, TimeTensor

    data = np.random.randn(16, 50, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.random.randint(0, 3, 16))
    pipe = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    result = pipe.fit_transform(ts)
    assert result.shape == (16, 200)
    assert result.y is not None


test("Pipeline: fit_transform, y preserved", test_pipeline_fit_transform)


def test_pipeline_fit_then_transform():
    """fit on train data, transform on test data — stats learned from train only."""
    from tensorts import Dim, Flatten, Pipeline, StandardScaler, TimeTensor, temporal_split

    data = np.random.randn(100, 50, 4) * 10
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.random.randint(0, 3, 100))
    train, test = temporal_split(ts, test_ratio=0.2)

    pipe = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    Xt_train = pipe.fit_transform(train)
    Xt_test = pipe.transform(test)  # uses train statistics

    assert Xt_train.shape == (80, 200)
    assert Xt_test.shape == (20, 200)
    # Both have same feature count
    assert Xt_train.shape[1] == Xt_test.shape[1]


test("Pipeline: fit on train, transform test (no leakage)", test_pipeline_fit_then_transform)


def test_pipeline_get_set_params():
    from tensorts import Flatten, Pipeline, StandardScaler

    pipe = Pipeline([("scale", StandardScaler()), ("flat", Flatten())])
    params = pipe.get_params(deep=True)
    assert "scale__eps" in params
    pipe.set_params(scale__eps=1e-4)
    assert pipe["scale"].eps == 1e-4


test("Pipeline: nested get_params / set_params", test_pipeline_get_set_params)


def test_channel_transform_per_group():
    from tensorts import ChannelTransform, Dim, MinMaxScaler, StandardScaler, TimeTensor

    data = np.random.randn(8, 100, 4) * np.array([1, 10, 100, 1000])
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c", "d"],
        channel_groups={"ab": ["a", "b"], "cd": ["c", "d"]},
    )
    ct = ChannelTransform({"ab": StandardScaler(), "cd": MinMaxScaler()})
    result = ct.fit_transform(ts)
    assert result.n_channels == 4
    # ab channels should be ~zero mean (StandardScaler)
    ab = result.ch["a", "b"]
    assert abs(np.mean(ab.data)) < 0.2
    # cd channels should be in [0,1] (MinMaxScaler)
    cd = result.ch["c", "d"]
    assert cd.data.min() >= -0.01
    assert cd.data.max() <= 1.01


test("ChannelTransform: different scalers per group", test_channel_transform_per_group)


def test_channel_transform_remainder_drop():
    from tensorts import ChannelTransform, Dim, StandardScaler, TimeTensor

    data = np.random.randn(4, 50, 4)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c", "d"],
        channel_groups={"ab": ["a", "b"]},
    )
    ct = ChannelTransform({"ab": StandardScaler()}, remainder="drop")
    result = ct.fit_transform(ts)
    # Only 'ab' channels survive
    assert result.n_channels == 2
    assert result.channels.names == ("a", "b")


test("ChannelTransform: remainder='drop' removes unmatched channels", test_channel_transform_remainder_drop)


def test_channel_select():
    from tensorts import ChannelSelect, Dim, StandardScaler, TimeTensor

    data = np.random.randn(8, 100, 6) * 10
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"],
    )
    cs = ChannelSelect("a_*", StandardScaler())
    result = cs.fit_transform(ts)
    assert result.n_channels == 6  # all channels present
    a_data = result.ch["a_*"].data
    assert abs(np.mean(a_data)) < 0.2  # a_* scaled
    g_data = result.ch["g_*"].data
    np.testing.assert_array_equal(g_data, ts.ch["g_*"].data)  # g_* unchanged


test("ChannelSelect: transform subset, passthrough others unchanged", test_channel_select)


# =====================================================================
# 9. END-TO-END PIPELINES (dense + ragged)
# =====================================================================
print()
print("=" * 60)
print("END-TO-END PIPELINES")
print("=" * 60)


def test_pipeline_air_quality_dense():
    """Dense: air quality sensor → impute → scale → window → statistical features."""
    from tensorts import (
        Dim,
        Interpolate,
        Pipeline,
        SlidingWindow,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    B, T, C = 2, 500, 3
    data = np.random.randn(B, T, C) * 10
    mask = np.random.random((B, T, C)) < 0.02
    data[mask] = np.nan

    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["PM2.5", "PM10", "O3"])

    pipe = Pipeline(
        [
            ("impute", Interpolate(method="linear")),
            ("scale", StandardScaler()),
            ("window", SlidingWindow(size=50, stride=25)),
            ("stats", make_statistical_extractor(["mean", "std", "min", "max"])),
        ]
    )
    result = pipe.fit_transform(ts)

    # Dense SlidingWindow → (B, K_windows, W, C).
    # After stats: (B, K_windows, 4, C).  FEATURE axis has 4 stats.
    assert result.shape[0] == B  # batch preserved
    assert result.shape[2] == 4  # 4 statistical features
    assert result.shape[3] == C  # channels preserved
    assert not np.isnan(result.data).any()
    print(f"    Air quality dense: {ts.shape} → {result.shape}")


test("Pipeline (dense): air quality → impute → scale → window → stats → flatten", test_pipeline_air_quality_dense)


def test_pipeline_har_ragged():
    """Ragged: HAR recordings → per-group scaler → window → multi-fn features."""
    import numpy as np

    from tensorts import (
        ChannelTransform,
        Flatten,
        Pipeline,
        RobustScaler,
        SlidingWindow,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    rng = np.random.default_rng(0)
    samples = [rng.standard_normal((rng.integers(150, 400), 6)) for _ in range(30)]
    labels = np.tile([0, 1, 2], 10)

    ts = TimeTensor.from_ragged(
        samples,
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        channel_groups={"acc": "acc_*", "gyro": "gyro_*"},
        y=labels,
    )

    pipe = Pipeline(
        [
            ("scale", ChannelTransform({"acc": StandardScaler(), "gyro": RobustScaler()})),
            ("window", SlidingWindow(size=64, stride=32, label_strategy="repeat", flatten_batch=True)),
            ("stats", make_statistical_extractor(["mean", "std", "rms", "zcr"])),
            ("flat", Flatten()),
        ]
    )
    result = pipe.fit_transform(ts)

    # 4 stats × 6 channels = 24 features
    assert result.shape[-1] == 24
    assert result.y is not None
    assert len(result.y) == result.n_samples
    print(f"    HAR ragged: {ts.n_samples} samples → {result.shape}")


test("Pipeline (ragged): HAR → per-group scale → window → multi-fn stats → flatten", test_pipeline_har_ragged)


def test_pipeline_audio_mel():
    """Dense: audio → pre-emphasis → mel spectrogram → log → scale."""
    from tensorts import (
        Dim,
        LogTransform,
        MelSpectrogram,
        Pipeline,
        PreEmphasis,
        StandardScaler,
        TimeTensor,
    )

    sr = 16000
    data = np.random.randn(4, sr, 1)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["audio"],
        y=np.array([0, 1, 0, 1]),
        metadata={"sampling_rate": sr},
    )

    pipe = Pipeline(
        [
            ("pe", PreEmphasis(coeff=0.97)),
            ("mel", MelSpectrogram(n_fft=512, hop_length=160, n_mels=64, sampling_rate=sr)),
            ("log", LogTransform()),
            ("scale", StandardScaler()),
        ]
    )
    result = pipe.fit_transform(ts)

    assert result.shape[0] == 4
    assert result.shape[1] == 64  # mel bins
    assert result.has_dim(Dim.FREQUENCY)
    np.testing.assert_array_equal(result.y, [0, 1, 0, 1])
    print(f"    Audio mel: {ts.shape} → {result.shape}")


test("Pipeline (dense): audio → PreEmphasis → MelSpectrogram → LogTransform → StandardScaler", test_pipeline_audio_mel)


def test_pipeline_feature_extractor_multi_fn():
    """FeatureExtractor multi-fn mode with windowed ragged data."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        Flatten,
        Pipeline,
        SlidingWindow,
        StandardScaler,
        TimeTensor,
    )

    def autocorr_lag1(x):
        if len(x) < 2:
            return np.float64(0.0)
        return np.float64(np.corrcoef(x[:-1], x[1:])[0, 1])

    samples = [np.random.randn(200 + i * 30, 4) for i in range(20)]
    y = np.random.randint(0, 3, 20)
    ts = TimeTensor.from_ragged(samples, y=y, channels=["a", "b", "c", "d"])

    fe = FeatureExtractor(
        functions=[("mean", np.mean), ("std", np.std), ("lag1", autocorr_lag1)],
        input_dim=Dim.TIME,
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("window", SlidingWindow(size=50, stride=25, label_strategy="repeat", flatten_batch=True)),
            ("fe", fe),
            ("flat", Flatten()),
        ]
    )
    result = pipe.fit_transform(ts)

    # 3 features × 4 channels = 12
    assert result.shape[-1] == 12
    assert result.y is not None
    print(f"    Multi-fn FE: {ts.n_samples} ragged → {result.shape}")


test("Pipeline (ragged): multi-fn FeatureExtractor in windowed pipeline", test_pipeline_feature_extractor_multi_fn)


# =====================================================================
# 10. SPLIT UTILITIES
# =====================================================================
print()
print("=" * 60)
print("SPLIT UTILITIES")
print("=" * 60)


def test_temporal_split():
    from tensorts import Dim, TimeTensor, temporal_split

    data = np.random.randn(100, 50, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.arange(100))
    train, test = temporal_split(ts, test_ratio=0.2)
    assert train.n_samples == 80
    assert test.n_samples == 20
    # Temporal order preserved
    np.testing.assert_array_equal(train.y, np.arange(80))
    np.testing.assert_array_equal(test.y, np.arange(80, 100))


test("temporal_split: 80/20, order preserved", test_temporal_split)


def test_temporal_split_ragged():
    from tensorts import TimeTensor, temporal_split

    samples = [np.random.randn(np.random.randint(50, 150), 2) for _ in range(50)]
    ts = TimeTensor.from_ragged(samples, y=np.arange(50))
    train, test = temporal_split(ts, test_ratio=0.2)
    assert train.n_samples == 40
    assert test.n_samples == 10
    assert train.is_ragged


test("temporal_split: ragged data, order preserved", test_temporal_split_ragged)


def test_stratified_temporal_split():
    from tensorts import Dim, TimeTensor, stratified_temporal_split

    data = np.random.randn(100, 50, 3)
    y = np.array([0] * 50 + [1] * 50)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=y)
    train, test = stratified_temporal_split(ts, test_ratio=0.2)
    # Each class contributes exactly 10 to test set
    assert np.sum(test.y == 0) == 10
    assert np.sum(test.y == 1) == 10


test("stratified_temporal_split: balanced 80/20", test_stratified_temporal_split)


def test_kfold_temporal():
    from tensorts import Dim, TimeTensor, kfold_temporal

    data = np.random.randn(50, 20, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.arange(50))
    folds = kfold_temporal(ts, n_folds=5)
    assert len(folds) == 5
    for train_f, test_f in folds:
        # No temporal leakage: all train indices < all test indices
        assert train_f.n_samples > 0
        assert test_f.n_samples > 0
        assert train_f.y.max() < test_f.y.min()


test("kfold_temporal: 5 folds, no future leakage", test_kfold_temporal)


# =====================================================================
# Summary
# =====================================================================
print()
print("=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
import sys

sys.exit(1 if FAIL > 0 else 0)
