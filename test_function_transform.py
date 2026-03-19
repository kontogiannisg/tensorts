"""
Tests for FunctionTransform and StatefulFunctionTransform.

Covers:
1.  PER_CHANNEL — scipy median filter (shape-preserving)
2.  PER_CHANNEL — scipy sosfiltfilt with fn_kwargs
3.  PER_CHANNEL — shape-changing (downsample via fn)
4.  PER_SAMPLE — custom multivariate feature extractor
5.  PER_SAMPLE — shape-preserving (normalize per sample)
6.  GLOBAL — element-wise clipping
7.  GLOBAL — full tensor manipulation
8.  Ragged + PER_CHANNEL
9.  Ragged + PER_SAMPLE
10. StatefulFunctionTransform — per-channel percentile clipping
11. StatefulFunctionTransform — global custom scaler
12. Pipeline integration — FunctionTransform in a full pipeline
13. ChannelSelect integration — wrap scipy filter on subset
14. 4D tensor — PER_CHANNEL on (B, K, T, C)
15. fn_args forwarding
"""

import numpy as np
import traceback

np.random.seed(42)

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗ {name}")
        traceback.print_exc()
        print()


# =====================================================================
# 1. PER_CHANNEL — stateless, shape-preserving
# =====================================================================
print("=" * 60)
print("FUNCTION TRANSFORM — PER_CHANNEL")
print("=" * 60)



# =====================================================================
# 2. PER_SAMPLE — stateless
# =====================================================================
print()
print("=" * 60)
print("FUNCTION TRANSFORM — PER_SAMPLE")
print("=" * 60)


def test_per_sample_feature_extractor():
    """PER_SAMPLE: custom function (T, C) → (K,) feature vector."""
    from tensorts import TimeTensor, FunctionTransform, Dim, Scope

    def extract_features(x):
        """x is (T, C), return (K,) feature vector."""
        return np.array([
            np.mean(x, axis=0).ravel(),        # C means
            np.std(x, axis=0).ravel(),         # C stds
            np.max(x, axis=0).ravel(),         # C maxes
        ]).ravel()  # 3*C features

    data = np.random.randn(10, 200, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c", "d"])

    feat = FunctionTransform(
        fn=extract_features,
        scope=Scope.PER_SAMPLE,
        output_dims=(Dim.BATCH, Dim.FEATURE),
        output_channels=[f"f_{i}" for i in range(12)],  # 3 * 4 channels
    )
    result = feat.fit_transform(ts)

    assert result.shape == (10, 12), f"Got {result.shape}"
    assert result.has_dim(Dim.FEATURE)
    assert len(result.channels.names) == 12

test("PER_SAMPLE: custom (T,C)→(K,) feature extractor", test_per_sample_feature_extractor)


def test_per_sample_normalize():
    """PER_SAMPLE shape-preserving: z-score normalize each sample independently."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    def zscore_sample(x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    data = np.random.randn(8, 100, 3) * np.array([1, 10, 100]) + np.array([5, 50, 500])
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    norm = FunctionTransform(fn=zscore_sample, scope=Scope.PER_SAMPLE)
    result = norm.fit_transform(ts)

    assert result.shape == (8, 100, 3)
    # Each sample should have ~zero mean per channel
    for i in range(8):
        for c in range(3):
            assert abs(np.mean(result.data[i, :, c])) < 0.1

test("PER_SAMPLE: z-score per sample (shape-preserving)", test_per_sample_normalize)


# =====================================================================
# 3. GLOBAL — stateless
# =====================================================================
print()
print("=" * 60)
print("FUNCTION TRANSFORM — GLOBAL")
print("=" * 60)


def test_global_clip():
    """GLOBAL: element-wise clipping."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 50, 3) * 10
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    clip = FunctionTransform(
        fn=lambda x: np.clip(x, -2, 2),
        scope=Scope.GLOBAL,
    )
    result = clip.fit_transform(ts)

    assert result.shape == (4, 50, 3)
    assert result.data.min() >= -2.0
    assert result.data.max() <= 2.0

test("GLOBAL: element-wise clipping", test_global_clip)


def test_global_power_spectrum():
    """GLOBAL: batch FFT (full tensor operation)."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 128, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    power = FunctionTransform(
        fn=lambda x: np.abs(np.fft.rfft(x, axis=1)) ** 2,
        scope=Scope.GLOBAL,
        output_dims=(Dim.BATCH, Dim.FREQUENCY, Dim.CHANNEL),
    )
    result = power.fit_transform(ts)

    assert result.shape == (4, 65, 2)  # rfft of 128 → 65 bins
    assert result.has_dim(Dim.FREQUENCY)

test("GLOBAL: batch FFT power spectrum", test_global_power_spectrum)


# =====================================================================
# 4. Ragged support
# =====================================================================
print()
print("=" * 60)
print("FUNCTION TRANSFORM — RAGGED")
print("=" * 60)


def test_ragged_per_channel():
    """PER_CHANNEL on ragged: apply scipy filter to variable-length samples."""
    from scipy.signal import medfilt
    from tensorts import TimeTensor, FunctionTransform, Scope

    samples = [np.random.randn(80, 2), np.random.randn(120, 2), np.random.randn(60, 2)]
    ts = TimeTensor.from_ragged(samples, channels=["x", "y"])

    smooth = FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                                fn_kwargs={"kernel_size": 3})
    result = smooth.fit_transform(ts)

    assert result.is_ragged
    assert len(result._samples) == 3
    assert result._samples[0].shape == (80, 2)
    assert result._samples[1].shape == (120, 2)
    assert result._samples[2].shape == (60, 2)

test("Ragged PER_CHANNEL: scipy medfilt on variable lengths", test_ragged_per_channel)


def test_ragged_per_sample():
    """PER_SAMPLE on ragged: extract features from variable-length samples."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    def extract(x):
        """(T, C) → (2*C,) features: mean and std per channel."""
        return np.concatenate([x.mean(axis=0), x.std(axis=0)])

    samples = [np.random.randn(80, 3), np.random.randn(120, 3), np.random.randn(60, 3)]
    y = np.array([0, 1, 0])
    ts = TimeTensor.from_ragged(samples, channels=["a", "b", "c"], y=y)

    feat = FunctionTransform(
        fn=extract,
        scope=Scope.PER_SAMPLE,
        output_dims=(Dim.BATCH, Dim.FEATURE),
        output_channels=[f"f_{i}" for i in range(6)],
    )
    result = feat.fit_transform(ts)

    # All samples produce same shape → should go dense
    assert not result.is_ragged
    assert result.shape == (3, 6)
    np.testing.assert_array_equal(result.y, y)

test("Ragged PER_SAMPLE: feature extraction → dense output", test_ragged_per_sample)


def test_ragged_per_sample_still_ragged():
    """PER_SAMPLE on ragged: output shapes differ → stays ragged."""
    from tensorts import TimeTensor, FunctionTransform, Scope

    def halve(x):
        """(T, C) → (T//2, C) — output length depends on input."""
        return x[::2]

    samples = [np.random.randn(80, 2), np.random.randn(120, 2), np.random.randn(61, 2)]
    ts = TimeTensor.from_ragged(samples, channels=["x", "y"])

    half = FunctionTransform(fn=halve, scope=Scope.PER_SAMPLE)
    result = half.fit_transform(ts)

    assert result.is_ragged
    assert result._samples[0].shape == (40, 2)
    assert result._samples[1].shape == (60, 2)
    assert result._samples[2].shape == (31, 2)

test("Ragged PER_SAMPLE: variable output → stays ragged", test_ragged_per_sample_still_ragged)


# =====================================================================
# 5. StatefulFunctionTransform
# =====================================================================
print()
print("=" * 60)
print("STATEFUL FUNCTION TRANSFORM")
print("=" * 60)


def test_stateful_global_scaler():
    """StatefulFunctionTransform GLOBAL: custom scaler with fit/transform."""
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope

    data = np.random.randn(16, 100, 4) * 10 + 5
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    scaler = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": x.mean(axis=(0, 1), keepdims=True),
                          "sigma": x.std(axis=(0, 1), keepdims=True)},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.GLOBAL,
    )
    result = scaler.fit_transform(ts)

    assert result.shape == (16, 100, 4)
    for c in range(4):
        assert abs(np.mean(result.data[:, :, c])) < 0.2

test("Stateful GLOBAL: custom mean/std scaler", test_stateful_global_scaler)


def test_stateful_per_channel_clip():
    """StatefulFunctionTransform PER_CHANNEL: adaptive clipping per channel."""
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope

    data = np.random.randn(8, 100, 3) * 10
    # Inject outliers
    data[0, 0, 0] = 500
    data[1, 1, 1] = -400
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    clipper = StatefulFunctionTransform(
        fit_fn=lambda x: {"lo": np.percentile(x, 1), "hi": np.percentile(x, 99)},
        transform_fn=lambda x, s: np.clip(x, s["lo"], s["hi"]),
        scope=Scope.PER_CHANNEL,
    )
    result = clipper.fit_transform(ts)

    assert result.shape == (8, 100, 3)
    # Outliers should be clipped
    assert result.data.max() < 500
    assert result.data.min() > -400

test("Stateful PER_CHANNEL: adaptive percentile clipping", test_stateful_per_channel_clip)


def test_stateful_per_channel_ragged():
    """StatefulFunctionTransform PER_CHANNEL on ragged data."""
    from tensorts import TimeTensor, StatefulFunctionTransform, Scope

    samples = [np.random.randn(80, 2) * 5 + 10 for _ in range(10)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])

    scaler = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": np.mean(x), "sigma": np.std(x)},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.PER_CHANNEL,
    )
    result = scaler.fit_transform(ts)

    assert result.is_ragged
    all_data = np.concatenate(result._samples, axis=0)
    assert abs(np.mean(all_data[:, 0])) < 0.3
    assert abs(np.mean(all_data[:, 1])) < 0.3

test("Stateful PER_CHANNEL: ragged data", test_stateful_per_channel_ragged)


def test_stateful_train_test_split():
    """Fit StatefulFunctionTransform on train, apply on test."""
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope, temporal_split

    data = np.random.randn(50, 100, 2) * 10 + 5
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), y=np.arange(50))

    train, test_ts = temporal_split(ts, test_ratio=0.2)

    scaler = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": x.mean(axis=(0, 1), keepdims=True),
                          "sigma": x.std(axis=(0, 1), keepdims=True)},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.GLOBAL,
    )

    scaler.fit(train)
    result_train = scaler.transform(train)
    result_test = scaler.transform(test_ts)

    assert result_train.shape == (40, 100, 2)
    assert result_test.shape == (10, 100, 2)
    # Both use train's statistics — test won't be perfectly zero-mean
    assert abs(np.mean(result_train.data)) < 0.2

test("Stateful: fit on train, transform test (no leakage)", test_stateful_train_test_split)


# =====================================================================
# 6. Pipeline & ChannelSelect integration
# =====================================================================
print()
print("=" * 60)
print("INTEGRATION TESTS")
print("=" * 60)


def test_pipeline_integration():
    """FunctionTransform inside a full Pipeline."""
    from scipy.signal import medfilt
    from tensorts import (
        TimeTensor, Dim, Pipeline, FunctionTransform, Scope,
        StandardScaler, SlidingWindow, StatisticalFeatures,
    )

    data = np.random.randn(20, 200, 4)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c", "d"],
        y=np.random.randint(0, 3, 20),
    )

    pipe = Pipeline([
        ("smooth",  FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                                       fn_kwargs={"kernel_size": 5})),
        ("scale",   StandardScaler()),
        ("window",  SlidingWindow(size=50, stride=25)),
        ("stats",   StatisticalFeatures(["mean", "std"])),
    ])
    result = pipe.fit_transform(ts)

    # 2 stats × 4 channels = 8 features per window
    assert result.shape[-1] == 8
    assert result.y is not None

test("Pipeline: FunctionTransform + StandardScaler + Window + Stats", test_pipeline_integration)


def test_channel_select_integration():
    """FunctionTransform inside ChannelSelect — filter only some channels."""
    from scipy.signal import medfilt
    from tensorts import (
        TimeTensor, Dim, ChannelSelect, FunctionTransform, Scope,
    )

    data = np.random.randn(4, 100, 6)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"],
    )

    cs = ChannelSelect(
        "a_*",
        FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                          fn_kwargs={"kernel_size": 5}),
    )
    result = cs.fit_transform(ts)

    assert result.n_channels == 6
    # a_* channels should be smoothed, g_* unchanged
    g_orig = ts.ch["g_*"].data
    g_result = result.ch["g_*"].data
    np.testing.assert_array_equal(g_result, g_orig)

test("ChannelSelect: FunctionTransform on subset, passthrough others", test_channel_select_integration)


def test_4d_per_channel():
    """PER_CHANNEL on 4D tensor (B, K, T, C)."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 3, 50, 2)
    ts = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL),
        channels=["x", "y"],
    )

    smooth = FunctionTransform(
        fn=lambda x: np.convolve(x, np.ones(3)/3, mode="same"),
        scope=Scope.PER_CHANNEL,
    )
    result = smooth.fit_transform(ts)

    assert result.shape == (4, 3, 50, 2), f"Got {result.shape}"

test("PER_CHANNEL on 4D (B,K,T,C) tensor", test_4d_per_channel)


def test_pipeline_stateful_and_stateless():
    """Mix StatefulFunctionTransform and FunctionTransform in a pipeline."""
    from tensorts import (
        TimeTensor, Dim, Pipeline, FunctionTransform,
        StatefulFunctionTransform, Scope, Flatten, temporal_split,
    )

    data = np.random.randn(30, 100, 3) * np.array([1, 10, 100])
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c"], y=np.random.randint(0, 2, 30))

    train, test_ts = temporal_split(ts, test_ratio=0.2)

    pipe = Pipeline([
        ("clip", FunctionTransform(
            fn=lambda x: np.clip(x, -3, 3),
            scope=Scope.PER_CHANNEL,
        )),
        ("scale", StatefulFunctionTransform(
            fit_fn=lambda x: {"mu": np.mean(x), "sigma": np.std(x)},
            transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
            scope=Scope.PER_CHANNEL,
        )),
        ("flatten", Flatten()),
    ])

    X_train = pipe.fit_transform(train)
    X_test = pipe.transform(test_ts)

    assert X_train.shape[-1] == X_test.shape[-1]
    assert X_train.shape[-1] == 100 * 3  # T * C flattened

test("Pipeline: mixed stateful + stateless FunctionTransforms", test_pipeline_stateful_and_stateless)


def test_lambda_naming():
    """Verify repr shows useful name."""
    from tensorts import FunctionTransform, StatefulFunctionTransform, Scope

    def my_cool_filter(x):
        return x * 2

    ft = FunctionTransform(fn=my_cool_filter, scope=Scope.PER_CHANNEL)
    assert "my_cool_filter" in repr(ft)

    ft2 = FunctionTransform(fn=lambda x: x, scope=Scope.GLOBAL, name="identity")
    assert "identity" in repr(ft2)

    sft = StatefulFunctionTransform(
        fit_fn=lambda x: {}, transform_fn=lambda x, s: x,
        scope=Scope.GLOBAL, name="noop_stateful",
    )
    assert "noop_stateful" in repr(sft)

test("Repr shows function name or custom name", test_lambda_naming)


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
