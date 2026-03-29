"""
Tests for FeatureUnion — parallel feature extraction with heterogeneous scopes.

FeatureUnion runs multiple transforms on the same input and concatenates
their flattened outputs along a FEATURE axis. This enables combining
per-channel statistics with per-sample cross-channel features in a single
pipeline.
"""

import traceback

import numpy as np

np.random.seed(0)

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
# PART 1: Basic FeatureUnion
# =====================================================================
print("=" * 60)
print("PART 1: Basic FeatureUnion")
print("=" * 60)


def test_two_extractors_dense():
    """Two multi-fn extractors combined, both per-channel."""
    from tensorts import Dim, FeatureUnion, TimeTensor, make_statistical_extractor

    data = np.random.randn(8, 100, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["x", "y", "z"])

    fu = FeatureUnion(
        [
            ("stats1", make_statistical_extractor(["mean", "std"])),
            ("stats2", make_statistical_extractor(["min", "max"])),
        ]
    )
    result = fu.fit_transform(ts)

    # 2 stats × 3 ch + 2 stats × 3 ch = 12
    assert result.shape == (8, 12), f"Expected (8, 12), got {result.shape}"
    assert result.has_dim(Dim.FEATURE)
    assert len(result.channels.names) == 12
    print(f"    shape: {result.shape}, features: {result.channels.names[:4]}...")


test("Two per-channel extractors", test_two_extractors_dense)


def test_mixed_scope_per_channel_and_per_sample():
    """Core use case: per-channel stats + per-sample cross-channel features."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Scope,
        TimeTensor,
        make_statistical_extractor,
    )

    def cross_channel_features(x):
        sma = float(np.mean(np.sum(np.abs(x), axis=1)))
        mag = float(np.mean(np.linalg.norm(x, axis=1)))
        return np.array([sma, mag])

    data = np.random.randn(10, 128, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["ax", "ay", "az"])

    fu = FeatureUnion(
        [
            ("stats", make_statistical_extractor(["mean", "std"])),
            (
                "cross",
                FeatureExtractor(
                    fn=cross_channel_features,
                    scope=Scope.PER_SAMPLE,
                    name="cross",
                ),
            ),
        ]
    )
    result = fu.fit_transform(ts)

    # stats: 2 × 3 = 6, cross: 2 → total 8
    assert result.shape == (10, 8), f"Expected (10, 8), got {result.shape}"
    names = list(result.channels.names)
    assert len(names) == 8
    # Cross features should use the function name
    assert any("cross" in n for n in names)
    print(f"    shape: {result.shape}, features: {names}")


test("Mixed scope: per-channel + per-sample", test_mixed_scope_per_channel_and_per_sample)


def test_single_transformer():
    """FeatureUnion with one transformer should still work."""
    from tensorts import Dim, FeatureUnion, TimeTensor, make_statistical_extractor

    data = np.random.randn(5, 50, 2)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b"])

    fu = FeatureUnion([("stats", make_statistical_extractor(["mean"]))])
    result = fu.fit_transform(ts)

    assert result.shape == (5, 2), f"Expected (5, 2), got {result.shape}"


test("Single transformer in FeatureUnion", test_single_transformer)


def test_y_preserved():
    """Labels should pass through FeatureUnion."""
    from tensorts import Dim, FeatureUnion, TimeTensor, make_statistical_extractor

    data = np.random.randn(6, 50, 2)
    y = np.array([0, 1, 2, 0, 1, 2])
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b"], y=y)

    fu = FeatureUnion([("stats", make_statistical_extractor(["mean", "std"]))])
    result = fu.fit_transform(ts)

    np.testing.assert_array_equal(result.y, y)


test("Labels preserved through FeatureUnion", test_y_preserved)


# =====================================================================
# PART 2: Pipeline integration
# =====================================================================
print()
print("=" * 60)
print("PART 2: Pipeline integration")
print("=" * 60)


def test_pipeline_with_feature_union():
    """FeatureUnion inside a Pipeline: scaler → union → flatten."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Pipeline,
        Scope,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    def acc_magnitude_stats(x):
        mag = np.linalg.norm(x, axis=1)
        return np.array([np.mean(mag), np.std(mag)])

    data = np.random.randn(12, 128, 6)
    y = np.random.randint(0, 3, 12)
    ts = TimeTensor(
        data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["ax", "ay", "az", "gx", "gy", "gz"],
        y=y,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean", "std", "rms"])),
                        (
                            "cross",
                            FeatureExtractor(
                                fn=acc_magnitude_stats,
                                scope=Scope.PER_SAMPLE,
                                name="mag",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    result = pipe.fit_transform(ts)

    # stats: 3 × 6 = 18, cross: 2 → total 20
    assert result.shape == (12, 20), f"Expected (12, 20), got {result.shape}"
    np.testing.assert_array_equal(result.y, y)
    print(f"    Pipeline shape: {result.shape}")


test("Pipeline: Scaler → FeatureUnion", test_pipeline_with_feature_union)


def test_pipeline_fit_then_transform():
    """Separate fit and transform calls work correctly."""
    from tensorts import (
        Dim,
        FeatureUnion,
        Pipeline,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    data_train = np.random.randn(10, 100, 3)
    data_test = np.random.randn(5, 100, 3)
    ts_train = TimeTensor(data_train, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"])
    ts_test = TimeTensor(data_test, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean", "std"])),
                    ]
                ),
            ),
        ]
    )
    pipe.fit(ts_train)
    result = pipe.transform(ts_test)

    assert result.shape == (5, 6), f"Expected (5, 6), got {result.shape}"


test("Pipeline: separate fit/transform", test_pipeline_fit_then_transform)


def test_sklearn_wrapper_with_feature_union():
    """SklearnWrapper with FeatureUnion works with cross_val_score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Pipeline,
        Scope,
        SklearnWrapper,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    def total_energy(x):
        return float(np.sum(x**2))

    data = np.random.randn(30, 64, 3)
    y = np.random.randint(0, 2, 30)
    ts = TimeTensor(
        data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c"],
        y=y,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean", "std"])),
                        (
                            "cross",
                            FeatureExtractor(
                                fn=total_energy,
                                scope=Scope.PER_SAMPLE,
                                output_dims=(),
                                name="energy",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    wrapper = SklearnWrapper(pipe, LogisticRegression(max_iter=200))
    scores = cross_val_score(wrapper, ts, y=ts.y, cv=3)

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    print(f"    cross_val_score: {scores}")


test("SklearnWrapper + FeatureUnion + cross_val_score", test_sklearn_wrapper_with_feature_union)


# =====================================================================
# PART 3: Edge cases and validation
# =====================================================================
print()
print("=" * 60)
print("PART 3: Edge cases and validation")
print("=" * 60)


def test_empty_transformers_raises():
    """Empty transformer list should raise ValueError."""
    from tensorts import FeatureUnion

    try:
        FeatureUnion([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


test("Empty transformers raises ValueError", test_empty_transformers_raises)


def test_duplicate_names_raises():
    """Duplicate transformer names should raise ValueError."""
    from tensorts import FeatureUnion, make_statistical_extractor

    try:
        FeatureUnion(
            [
                ("stats", make_statistical_extractor(["mean"])),
                ("stats", make_statistical_extractor(["std"])),
            ]
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


test("Duplicate names raises ValueError", test_duplicate_names_raises)


def test_get_set_params():
    """get_params/set_params work for sklearn clone compatibility."""
    from sklearn.base import clone

    from tensorts import FeatureUnion, make_statistical_extractor

    fu = FeatureUnion(
        [
            ("s1", make_statistical_extractor(["mean"])),
            ("s2", make_statistical_extractor(["std"])),
        ]
    )
    params = fu.get_params(deep=True)
    assert "transformers" in params
    assert "s1" in params
    assert "s2" in params

    cloned = clone(fu)
    assert len(cloned.transformers) == 2


test("get_params / set_params / clone", test_get_set_params)


def test_partial_fit():
    """partial_fit works through FeatureUnion."""
    from tensorts import Dim, FeatureUnion, TimeTensor, make_statistical_extractor

    data1 = np.random.randn(5, 50, 2)
    data2 = np.random.randn(5, 50, 2)
    ts1 = TimeTensor(data1, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b"])
    ts2 = TimeTensor(data2, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b"])

    fu = FeatureUnion([("stats", make_statistical_extractor(["mean", "std"]))])
    fu.partial_fit(ts1)
    fu.partial_fit(ts2)
    result = fu.transform(ts2)

    assert result.shape == (5, 4)


test("partial_fit through FeatureUnion", test_partial_fit)


def test_repr():
    """repr should list transformer names."""
    from tensorts import FeatureUnion, make_statistical_extractor

    fu = FeatureUnion(
        [
            ("time_stats", make_statistical_extractor(["mean"])),
            ("freq_stats", make_statistical_extractor(["std"])),
        ]
    )
    r = repr(fu)
    assert "time_stats" in r
    assert "freq_stats" in r


test("repr lists transformer names", test_repr)


# =====================================================================
# PART 4: Real-world HAR simulation
# =====================================================================
print()
print("=" * 60)
print("PART 4: Real-world HAR simulation")
print("=" * 60)


def test_har_full_pipeline():
    """Simulated HAR pipeline: scaler → FeatureUnion(stats + cross) → classify."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        LabelEncoder,
        Pipeline,
        Scope,
        SklearnWrapper,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )

    def cross_channel_features(x):
        sma_acc = float(np.mean(np.sum(np.abs(x[:, 0:3]), axis=1)))
        sma_gyro = float(np.mean(np.sum(np.abs(x[:, 3:6]), axis=1)))
        acc_mag = np.linalg.norm(x[:, 0:3], axis=1)
        return np.array([sma_acc, sma_gyro, np.mean(acc_mag), np.std(acc_mag)])

    # Simulated 6-axis IMU data (acc + gyro), 50 samples × 128 timesteps
    data = np.random.randn(50, 128, 6)
    y = np.random.randint(0, 6, 50)
    ts = TimeTensor(
        data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["ax", "ay", "az", "gx", "gy", "gz"],
        y=y,
    )

    pipe = Pipeline(
        [
            ("label_enc", LabelEncoder()),
            ("scaler", StandardScaler()),
            (
                "features",
                FeatureUnion(
                    [
                        ("time", make_statistical_extractor(["mean", "std", "rms", "zcr"])),
                        (
                            "cross",
                            FeatureExtractor(
                                fn=cross_channel_features,
                                scope=Scope.PER_SAMPLE,
                                name="cross",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    wrapper = SklearnWrapper(pipe, RandomForestClassifier(n_estimators=10, random_state=42))
    scores = cross_val_score(wrapper, ts, y=ts.y, cv=3)

    # time: 4 stats × 6 ch = 24, cross: 4 → total 28
    wrapper.fit(ts, y=ts.y)
    result = pipe.transform(ts)
    assert result.shape == (50, 28), f"Expected (50, 28), got {result.shape}"
    print(f"    HAR pipeline: {ts.shape} → {result.shape}")
    print(f"    cross_val_score: {scores}")
    print(f"    feature names: {result.channels.names[:6]}...")


test("HAR pipeline: Scaler → FeatureUnion(stats+cross) → RF", test_har_full_pipeline)


# =====================================================================
# PART 5: 4D input (SlidingWindow + FeatureUnion)
# =====================================================================
print()
print("=" * 60)
print("PART 5: 4D input (SlidingWindow + FeatureUnion)")
print("=" * 60)


def test_4d_two_extractors():
    """Two per-channel extractors after SlidingWindow(flatten_batch=False)."""
    from tensorts import Dim, FeatureUnion, Pipeline, StandardScaler, TimeTensor, make_statistical_extractor
    from tensorts.transforms.temporal import SlidingWindow

    data = np.random.randn(4, 32, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["x", "y", "z"])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("window", SlidingWindow(size=5, stride=2, flatten_batch=False)),
            (
                "features",
                FeatureUnion(
                    [
                        ("s1", make_statistical_extractor(["mean", "std"])),
                        ("s2", make_statistical_extractor(["rms"])),
                    ]
                ),
            ),
        ]
    )
    result = pipe.fit_transform(ts)

    assert Dim.WINDOW in result.dims, f"WINDOW dim not preserved: {result.dims}"
    # 14 windows, (2+1) stats × 3 channels = 9
    assert result.shape == (4, 14, 9), f"Expected (4, 14, 9), got {result.shape}"
    assert result.dims == (Dim.BATCH, Dim.WINDOW, Dim.FEATURE)
    names = list(result.channels.names)
    assert len(names) == 9
    assert "mean__x" in names
    print(f"    shape: {result.shape}, features: {names[:4]}...")


test("4D: two per-channel extractors", test_4d_two_extractors)


def test_4d_mixed_scope():
    """4D: per-channel stats + per-sample cross-channel features."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Pipeline,
        Scope,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )
    from tensorts.transforms.temporal import SlidingWindow

    def cross_features(x):
        mag = float(np.mean(np.linalg.norm(x, axis=1)))
        sma = float(np.mean(np.sum(np.abs(x), axis=1)))
        return np.array([mag, sma])

    data = np.random.randn(4, 32, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("window", SlidingWindow(size=5, stride=2, flatten_batch=False)),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean"])),
                        (
                            "cross",
                            FeatureExtractor(
                                fn=cross_features,
                                scope=Scope.PER_SAMPLE,
                                name="cross",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    result = pipe.fit_transform(ts)

    # 1 stat × 3 ch + 2 cross = 5 per window
    assert result.shape == (4, 14, 5), f"Expected (4, 14, 5), got {result.shape}"
    names = list(result.channels.names)
    assert "cross_0" in names
    print(f"    shape: {result.shape}, features: {names}")


test("4D: mixed scope per-channel + per-sample", test_4d_mixed_scope)


def test_4d_scalar_per_sample():
    """4D: scalar PER_SAMPLE output alongside per-channel stats."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Pipeline,
        Scope,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )
    from tensorts.transforms.temporal import SlidingWindow

    def total_energy(x):
        return float(np.sum(x**2))

    data = np.random.randn(4, 32, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["a", "b", "c"])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("window", SlidingWindow(size=5, stride=2, flatten_batch=False)),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean"])),
                        (
                            "energy",
                            FeatureExtractor(
                                fn=total_energy,
                                scope=Scope.PER_SAMPLE,
                                output_dims=(),
                                name="energy",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    result = pipe.fit_transform(ts)

    # 1 × 3 + 1 = 4 per window
    assert result.shape == (4, 14, 4), f"Expected (4, 14, 4), got {result.shape}"
    assert "energy" in list(result.channels.names)
    print(f"    shape: {result.shape}, features: {list(result.channels.names)}")


test("4D: scalar PER_SAMPLE + per-channel stats", test_4d_scalar_per_sample)


def test_4d_multi_fn_per_sample():
    """4D: multi-fn PER_SAMPLE alongside per-channel stats."""
    from tensorts import (
        Dim,
        FeatureExtractor,
        FeatureUnion,
        Pipeline,
        Scope,
        StandardScaler,
        TimeTensor,
        make_statistical_extractor,
    )
    from tensorts.transforms.temporal import SlidingWindow

    def mag_fn(x):
        return float(np.linalg.norm(x).mean())

    def sma_fn(x):
        return float(np.sum(np.abs(x)))

    data = np.random.randn(4, 32, 6)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=["ax", "ay", "az", "gx", "gy", "gz"])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("window", SlidingWindow(size=5, stride=2, flatten_batch=False)),
            (
                "features",
                FeatureUnion(
                    [
                        ("stats", make_statistical_extractor(["mean"])),
                        (
                            "cross",
                            FeatureExtractor(
                                functions=[("mag", mag_fn), ("sma", sma_fn)],
                                scope=Scope.PER_SAMPLE,
                                name="cross",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    result = pipe.fit_transform(ts)

    # 1 × 6 + 2 = 8 per window
    assert result.shape == (4, 14, 8), f"Expected (4, 14, 8), got {result.shape}"
    names = list(result.channels.names)
    assert "mag" in names
    assert "sma" in names
    print(f"    shape: {result.shape}, features: {names}")


test("4D: multi-fn PER_SAMPLE + per-channel stats", test_4d_multi_fn_per_sample)


def test_4d_incompatible_shapes_raises():
    """Mixing windowed and non-windowed sub-transforms should raise."""
    from tensorts import Dim, FeatureUnion

    try:
        FeatureUnion._validate_leading(
            [
                ("stats", (4, 14), (Dim.BATCH, Dim.WINDOW)),
                ("cross", (4,), (Dim.BATCH,)),
            ]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "incompatible leading shapes" in str(e)


test("4D: incompatible leading shapes raises ValueError", test_4d_incompatible_shapes_raises)


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
