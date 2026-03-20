"""
Tests for FunctionTransform and StatefulFunctionTransform.

Design rules this file verifies:
  - FunctionTransform wraps stateless functions; dim schema is PRESERVED.
    fn receives the appropriate slice and returns the same rank (ndim) but
    may change size along any axis.
  - StatefulFunctionTransform separates fit (learn state) from transform
    (apply state); separate states are learned per Scope.
  - Neither class changes dim semantics — use FeatureExtractor for that.

Scopes tested:
  PER_CHANNEL  fn(T,) → (T',)    — vectorized over batch and channels
  PER_SAMPLE   fn(T, C) → (T', C) — vectorized over batch
  GLOBAL       fn(full tensor) → same-rank result

Dense and ragged inputs are both tested for PER_CHANNEL and PER_SAMPLE.
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
# 1. FunctionTransform — PER_CHANNEL
# =====================================================================
print("=" * 60)
print("FunctionTransform — PER_CHANNEL")
print("=" * 60)


def test_per_channel_medfilt_dense():
    """PER_CHANNEL: scipy medfilt on each channel, shape unchanged."""
    from scipy.signal import medfilt
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(8, 100, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c", "d"])

    smooth = FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                                fn_kwargs={"kernel_size": 5})
    result = smooth.fit_transform(ts)

    assert result.shape == (8, 100, 4)
    assert result.dims == ts.dims            # dim schema preserved
    assert result.channels.names == ts.channels.names

test("PER_CHANNEL dense: scipy medfilt, shape + dims preserved", test_per_channel_medfilt_dense)


def test_per_channel_shape_changing_dense():
    """PER_CHANNEL: downsample — time axis changes length, ndim stays same."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(6, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    downsample = FunctionTransform(fn=lambda x: x[::2], scope=Scope.PER_CHANNEL)
    result = downsample.fit_transform(ts)

    assert result.shape == (6, 50, 3)        # TIME halved
    assert result.dims == ts.dims            # schema unchanged (TIME still TIME)

test("PER_CHANNEL dense: fn changes time length, ndim preserved", test_per_channel_shape_changing_dense)


def test_per_channel_4d():
    """PER_CHANNEL on (B, K, T, C) — operates on TIME axis, leaves WINDOW intact."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 5, 64, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL))

    smooth = FunctionTransform(
        fn=lambda x: np.convolve(x, np.ones(3) / 3, mode="same"),
        scope=Scope.PER_CHANNEL,
    )
    result = smooth.fit_transform(ts)
    assert result.shape == (4, 5, 64, 3)

test("PER_CHANNEL on 4D (B,K,T,C): operates on TIME, WINDOW preserved", test_per_channel_4d)


def test_per_channel_fn_kwargs():
    """PER_CHANNEL: extra kwargs forwarded to fn correctly."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    def band_limit(x, n_coeffs=10):
        fft = np.fft.rfft(x)
        fft[n_coeffs:] = 0
        return np.fft.irfft(fft, n=len(x))

    data = np.random.randn(4, 64, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    ft = FunctionTransform(fn=band_limit, scope=Scope.PER_CHANNEL,
                           fn_kwargs={"n_coeffs": 5})
    result = ft.fit_transform(ts)
    assert result.shape == (4, 64, 2)

test("PER_CHANNEL: fn_kwargs forwarded correctly", test_per_channel_fn_kwargs)


def test_per_channel_ragged():
    """PER_CHANNEL on ragged: each sample has different time length."""
    from scipy.signal import medfilt
    from tensorts import TimeTensor, FunctionTransform, Scope

    samples = [np.random.randn(80, 2), np.random.randn(120, 2), np.random.randn(60, 2)]
    y = np.array([0, 1, 0])
    ts = TimeTensor.from_ragged(samples, channels=["x", "y"], y=y)

    smooth = FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                                fn_kwargs={"kernel_size": 3})
    result = smooth.fit_transform(ts)

    assert result.is_ragged
    assert result._samples[0].shape == (80, 2)
    assert result._samples[1].shape == (120, 2)
    assert result._samples[2].shape == (60, 2)
    # Labels preserved
    np.testing.assert_array_equal(result.y, y)

test("PER_CHANNEL ragged: variable lengths preserved, y propagated", test_per_channel_ragged)


# =====================================================================
# 2. FunctionTransform — PER_SAMPLE
# =====================================================================
print()
print("=" * 60)
print("FunctionTransform — PER_SAMPLE")
print("=" * 60)

# NOTE: PER_SAMPLE fn receives (T_merged, C) and must return same ndim.
# It is designed for per-sample preprocessing, not dimensionality reduction.
# Use FeatureExtractor for (T, C) → (K,) feature extraction.


def test_per_sample_zscore_dense():
    """PER_SAMPLE: z-score each sample independently — (T,C) → (T,C)."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    def zscore(x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    data = np.random.randn(8, 100, 3) * np.array([1, 10, 100]) + np.array([5, 50, 500])
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    norm = FunctionTransform(fn=zscore, scope=Scope.PER_SAMPLE)
    result = norm.fit_transform(ts)

    assert result.shape == (8, 100, 3)
    # Each sample should have ~zero mean per channel
    for i in range(8):
        for c in range(3):
            assert abs(np.mean(result.data[i, :, c])) < 0.1

test("PER_SAMPLE dense: per-sample z-score, (T,C) → (T,C)", test_per_sample_zscore_dense)


def test_per_sample_clip_dense():
    """PER_SAMPLE: clip each sample to its own 1st–99th percentile."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    def adaptive_clip(x):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip(x, lo, hi)

    data = np.random.randn(6, 80, 2) * 5
    data[0, 0, 0] = 1000   # outlier in sample 0
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    clipper = FunctionTransform(fn=adaptive_clip, scope=Scope.PER_SAMPLE)
    result = clipper.fit_transform(ts)

    assert result.shape == (6, 80, 2)
    assert result.data.max() < 1000

test("PER_SAMPLE dense: per-sample adaptive clip, outlier removed", test_per_sample_clip_dense)


def test_per_sample_time_trim():
    """PER_SAMPLE: trim last 10 time steps — output shorter in TIME."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(5, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    trim = FunctionTransform(fn=lambda x: x[:-10], scope=Scope.PER_SAMPLE)
    result = trim.fit_transform(ts)

    assert result.shape == (5, 90, 3)   # TIME shortened
    assert result.dims == ts.dims        # dim schema preserved

test("PER_SAMPLE dense: trim time steps, ndim preserved", test_per_sample_time_trim)


def test_per_sample_ragged_same_output():
    """PER_SAMPLE ragged: all outputs same shape → dense output."""
    from tensorts import TimeTensor, FunctionTransform, Scope

    def normalize(x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    samples = [np.random.randn(80, 3) * 10, np.random.randn(80, 3) * 5,
               np.random.randn(80, 3)]  # all same (80, 3)
    y = np.array([0, 1, 2])
    ts = TimeTensor.from_ragged(samples, y=y)

    result = FunctionTransform(fn=normalize, scope=Scope.PER_SAMPLE).fit_transform(ts)

    # All same shape → stacked to dense
    assert not result.is_ragged
    assert result.shape == (3, 80, 3)
    np.testing.assert_array_equal(result.y, y)

test("PER_SAMPLE ragged (same shape): stacks to dense, y preserved",
     test_per_sample_ragged_same_output)


def test_per_sample_ragged_variable_output():
    """PER_SAMPLE ragged: output shapes vary → stays ragged."""
    from tensorts import TimeTensor, FunctionTransform, Scope

    def halve(x):
        return x[::2]  # output length = ceil(T/2), varies per sample

    samples = [np.random.randn(80, 2), np.random.randn(120, 2), np.random.randn(61, 2)]
    ts = TimeTensor.from_ragged(samples)

    result = FunctionTransform(fn=halve, scope=Scope.PER_SAMPLE).fit_transform(ts)

    assert result.is_ragged
    assert result._samples[0].shape == (40, 2)
    assert result._samples[1].shape == (60, 2)
    assert result._samples[2].shape == (31, 2)

test("PER_SAMPLE ragged (variable shape): stays ragged", test_per_sample_ragged_variable_output)


# =====================================================================
# 3. FunctionTransform — GLOBAL
# =====================================================================
print()
print("=" * 60)
print("FunctionTransform — GLOBAL")
print("=" * 60)


def test_global_clip():
    """GLOBAL: element-wise clipping via full tensor."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 50, 3) * 10
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    clip = FunctionTransform(fn=lambda x: np.clip(x, -2, 2), scope=Scope.GLOBAL)
    result = clip.fit_transform(ts)

    assert result.shape == (4, 50, 3)
    assert result.data.min() >= -2.0
    assert result.data.max() <=  2.0

test("GLOBAL: element-wise clipping", test_global_clip)


def test_global_batch_fft():
    """GLOBAL: vectorized FFT over entire (B, T, C) tensor."""
    from tensorts import TimeTensor, Dim, FunctionTransform, Scope

    data = np.random.randn(4, 128, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    power = FunctionTransform(
        fn=lambda x: np.abs(np.fft.rfft(x, axis=1)) ** 2,
        scope=Scope.GLOBAL,
    )
    result = power.fit_transform(ts)
    # rfft of 128 → 65 bins along TIME axis
    assert result.shape == (4, 65, 2)

test("GLOBAL: batch FFT power spectrum", test_global_batch_fft)


# =====================================================================
# 4. StatefulFunctionTransform
# =====================================================================
print()
print("=" * 60)
print("StatefulFunctionTransform")
print("=" * 60)


def test_stateful_global_scaler():
    """GLOBAL: learn mean/std from full tensor, apply in transform."""
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


def test_stateful_per_channel_adaptive_clip():
    """PER_CHANNEL: learn 1st–99th percentile per channel, clip in transform."""
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope

    data = np.random.randn(8, 100, 3) * 10
    data[0, 0, 0] = 500     # outlier ch 0
    data[1, 1, 1] = -400    # outlier ch 1
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    clipper = StatefulFunctionTransform(
        fit_fn=lambda x: {"lo": np.percentile(x, 1), "hi": np.percentile(x, 99)},
        transform_fn=lambda x, s: np.clip(x, s["lo"], s["hi"]),
        scope=Scope.PER_CHANNEL,
    )
    result = clipper.fit_transform(ts)

    assert result.shape == (8, 100, 3)
    assert result.data.max() < 500
    assert result.data.min() > -400

test("Stateful PER_CHANNEL: adaptive percentile clipping per channel", test_stateful_per_channel_adaptive_clip)


def test_stateful_per_channel_ragged():
    """PER_CHANNEL on ragged: state learned from concatenated samples."""
    from tensorts import TimeTensor, StatefulFunctionTransform, Scope

    samples = [np.random.randn(np.random.randint(60, 120), 2) * 5 + 10 for _ in range(10)]
    ts = TimeTensor.from_ragged(samples, channels=["a", "b"])

    scaler = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": np.mean(x), "sigma": np.std(x)},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.PER_CHANNEL,
    )
    result = scaler.fit_transform(ts)

    assert result.is_ragged
    all_data = np.concatenate(result._samples, axis=0)
    assert abs(np.mean(all_data[:, 0])) < 0.3    # channel 0 ~zero mean
    assert abs(np.mean(all_data[:, 1])) < 0.3    # channel 1 ~zero mean

test("Stateful PER_CHANNEL ragged: state learned across all samples", test_stateful_per_channel_ragged)


def test_stateful_fit_scope_vs_transform_scope():
    """fit_scope and scope can differ when both share the same state container.

    PER_SAMPLE fit (sets _state) + GLOBAL transform (reads _state) is a valid
    cross-scope combination — fit aggregates all samples, transform applies once.
    """
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope

    data = np.random.randn(10, 100, 4) * 10 + 5
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    sft = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": x.mean(), "sigma": x.std()},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.GLOBAL,
        fit_scope=Scope.PER_SAMPLE,    # fit_scope overrides fit granularity
    )
    result = sft.fit_transform(ts)
    assert result.shape == (10, 100, 4)

test("Stateful: fit_scope=PER_SAMPLE, scope=GLOBAL (shared _state)", test_stateful_fit_scope_vs_transform_scope)


def test_stateful_no_train_test_leakage():
    """Fit on train split, transform test — test statistics not used in fitting."""
    from tensorts import TimeTensor, Dim, StatefulFunctionTransform, Scope, temporal_split

    data = np.random.randn(50, 100, 2) * 10 + 5
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    y=np.arange(50))

    train, test = temporal_split(ts, test_ratio=0.2)

    scaler = StatefulFunctionTransform(
        fit_fn=lambda x: {"mu": x.mean(axis=(0, 1), keepdims=True),
                          "sigma": x.std(axis=(0, 1), keepdims=True)},
        transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
        scope=Scope.GLOBAL,
    )

    # Fit only on train
    scaler.fit(train)
    Xt_train = scaler.transform(train)
    Xt_test  = scaler.transform(test)       # uses train statistics

    assert Xt_train.shape == (40, 100, 2)
    assert Xt_test.shape  == (10, 100, 2)
    # Train should be well-normalized
    assert abs(np.mean(Xt_train.data)) < 0.2

test("Stateful: fit on train, transform test (no data leakage)", test_stateful_no_train_test_leakage)


# =====================================================================
# 5. Integration: Pipeline and ChannelSelect
# =====================================================================
print()
print("=" * 60)
print("INTEGRATION TESTS")
print("=" * 60)


def test_function_transform_in_pipeline():
    """FunctionTransform inside a Pipeline with other steps."""
    from scipy.signal import medfilt
    from tensorts import (
        TimeTensor, Dim, Pipeline, FunctionTransform, Scope,
        StandardScaler, SlidingWindow, Flatten, make_statistical_extractor,
    )

    data = np.random.randn(20, 200, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c", "d"],
                    y=np.random.randint(0, 3, 20))

    pipe = Pipeline([
        ("smooth", FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                                      fn_kwargs={"kernel_size": 5})),
        ("scale",  StandardScaler()),
        ("window", SlidingWindow(size=50, stride=25)),
        ("stats",  make_statistical_extractor(["mean", "std"])),
        ("flat",   Flatten()),
    ])
    result = pipe.fit_transform(ts)

    # Dense SlidingWindow → (B, K, W, C). After stats: (B, K, 2, C).
    # After flatten: (B, K * 2 * 4).  Check FEATURE axis has 2 stats × 4 channels.
    assert result.shape[0] == 20       # batch preserved
    assert result.shape[-1] % 8 == 0   # multiple of 2_stats × 4_channels
    assert result.y is not None

test("Pipeline: FunctionTransform + StandardScaler + SlidingWindow + multi-fn stats",
     test_function_transform_in_pipeline)


def test_function_transform_in_channel_select():
    """FunctionTransform inside ChannelSelect — smooth only some channels."""
    from scipy.signal import medfilt
    from tensorts import TimeTensor, Dim, ChannelSelect, FunctionTransform, Scope

    data = np.random.randn(4, 100, 6)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"])

    cs = ChannelSelect(
        "a_*",
        FunctionTransform(fn=medfilt, scope=Scope.PER_CHANNEL,
                          fn_kwargs={"kernel_size": 5}),
    )
    result = cs.fit_transform(ts)

    assert result.n_channels == 6
    # g_* channels are bit-for-bit identical to input
    np.testing.assert_array_equal(result.ch["g_*"].data, ts.ch["g_*"].data)

test("ChannelSelect: FunctionTransform on subset, passthrough others",
     test_function_transform_in_channel_select)


def test_mixed_stateful_stateless_pipeline():
    """Mix StatefulFunctionTransform and FunctionTransform in a pipeline."""
    from tensorts import (
        TimeTensor, Dim, Pipeline, FunctionTransform,
        StatefulFunctionTransform, Scope, Flatten, temporal_split,
    )

    data = np.random.randn(30, 100, 3) * np.array([1, 10, 100])
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c"], y=np.random.randint(0, 2, 30))

    train, test = temporal_split(ts, test_ratio=0.2)

    pipe = Pipeline([
        ("clip", FunctionTransform(fn=lambda x: np.clip(x, -3, 3),
                                   scope=Scope.PER_CHANNEL)),
        ("scale", StatefulFunctionTransform(
            fit_fn=lambda x: {"mu": np.mean(x), "sigma": np.std(x)},
            transform_fn=lambda x, s: (x - s["mu"]) / (s["sigma"] + 1e-8),
            scope=Scope.PER_CHANNEL,
        )),
        ("flat", Flatten()),
    ])

    Xt_train = pipe.fit_transform(train)
    Xt_test  = pipe.transform(test)

    assert Xt_train.shape[-1] == Xt_test.shape[-1]
    assert Xt_train.shape[-1] == 100 * 3     # T * C flattened

test("Pipeline: mixed stateful + stateless FunctionTransforms", test_mixed_stateful_stateless_pipeline)


def test_repr():
    """Repr shows function name or custom name."""
    from tensorts import FunctionTransform, StatefulFunctionTransform, Scope

    def my_filter(x):
        return x

    ft = FunctionTransform(fn=my_filter, scope=Scope.PER_CHANNEL)
    assert "my_filter" in repr(ft)

    ft2 = FunctionTransform(fn=lambda x: x, scope=Scope.GLOBAL, name="identity")
    assert "identity" in repr(ft2)

    sft = StatefulFunctionTransform(
        fit_fn=lambda x: {}, transform_fn=lambda x, s: x,
        scope=Scope.GLOBAL, name="noop",
    )
    assert "noop" in repr(sft)

test("Repr: function name / custom name appears in repr", test_repr)


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
