"""
Tests for partial_fit() — incremental learning support.

Covers:
  1. StandardScaler: Welford's online mean/variance vs full fit
  2. MinMaxScaler: running min/max vs full fit
  3. MaxAbsScaler: running max abs vs full fit
  4. LabelEncoder: append-only class mapping
  5. Interpolate: running channel means
  6. Pipeline.partial_fit: chain of transforms
  7. Stateless transforms: partial_fit works (no error)
  8. fit() resets state after partial_fit
  9. Ragged support for partial_fit
 10. SklearnWrapper.partial_fit with SGDClassifier
"""

import traceback

import numpy as np

from tensorts import (
    Dim,
    Flatten,
    MinMaxScaler,
    Pipeline,
    StandardScaler,
    TimeTensor,
)
from tensorts.transforms.preprocessing import (
    Interpolate,
    LabelEncoder,
    MaxAbsScaler,
    Normalizer,
    _extract_per_channel,
)

PASS = 0
FAIL = 0


def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS: {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL: {name}")
        traceback.print_exc()


# ======================================================================
# Helpers
# ======================================================================


def make_dense(B=20, T=50, C=3, seed=42):
    rng = np.random.RandomState(seed)
    data = rng.randn(B, T, C) * np.array([1.0, 10.0, 100.0])
    return TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c"],
    )


def make_ragged(n_samples=10, C=3, seed=42):
    rng = np.random.RandomState(seed)
    samples = [rng.randn(rng.randint(20, 80), C) * np.array([1.0, 10.0, 100.0]) for _ in range(n_samples)]
    return TimeTensor.from_ragged(samples, channels=["a", "b", "c"])


# ======================================================================
# StandardScaler
# ======================================================================

print("=" * 60)
print("partial_fit tests")
print("=" * 60)


def test_standard_scaler_equivalence():
    """partial_fit on two halves ≈ fit on full data."""
    X = make_dense(B=40, T=50, C=3)
    X1 = X[:20]
    X2 = X[20:]

    full = StandardScaler()
    full.fit(X)

    inc = StandardScaler()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.mean_, inc.mean_, atol=1e-10), f"mean mismatch: {full.mean_} vs {inc.mean_}"
    assert np.allclose(full.std_, inc.std_, atol=1e-10), f"std mismatch: {full.std_} vs {inc.std_}"
    assert inc.n_samples_seen_ == 40 * 50


run_test("StandardScaler partial_fit equivalence", test_standard_scaler_equivalence)


def test_standard_scaler_partial_fit_transform():
    """partial_fit + transform produces valid output."""
    X = make_dense()
    scaler = StandardScaler()
    scaler.partial_fit(X)
    Xt = scaler.transform(X)
    flat = _extract_per_channel(Xt)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-2)


run_test("StandardScaler partial_fit + transform", test_standard_scaler_partial_fit_transform)


def test_standard_scaler_ragged_partial_fit():
    """partial_fit works on ragged tensors."""
    X = make_ragged(n_samples=10)
    X1 = X[:5]
    X2 = X[5:]

    full = StandardScaler()
    full.fit(X)

    inc = StandardScaler()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.mean_, inc.mean_, atol=1e-10)
    assert np.allclose(full.std_, inc.std_, atol=1e-10)


run_test("StandardScaler ragged partial_fit", test_standard_scaler_ragged_partial_fit)


# ======================================================================
# MinMaxScaler
# ======================================================================


def test_minmax_scaler_equivalence():
    """partial_fit on two halves = fit on full data."""
    X = make_dense(B=40)
    X1 = X[:20]
    X2 = X[20:]

    full = MinMaxScaler()
    full.fit(X)

    inc = MinMaxScaler()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.min_, inc.min_), "min mismatch"
    assert np.allclose(full.max_, inc.max_), "max mismatch"


run_test("MinMaxScaler partial_fit equivalence", test_minmax_scaler_equivalence)


def test_minmax_scaler_ragged_partial_fit():
    """partial_fit on ragged tensors."""
    X = make_ragged(n_samples=10)
    X1 = X[:5]
    X2 = X[5:]

    full = MinMaxScaler()
    full.fit(X)

    inc = MinMaxScaler()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.min_, inc.min_)
    assert np.allclose(full.max_, inc.max_)


run_test("MinMaxScaler ragged partial_fit", test_minmax_scaler_ragged_partial_fit)


# ======================================================================
# MaxAbsScaler
# ======================================================================


def test_maxabs_scaler_equivalence():
    """partial_fit on two halves = fit on full data."""
    X = make_dense(B=40)
    X1 = X[:20]
    X2 = X[20:]

    full = MaxAbsScaler()
    full.fit(X)

    inc = MaxAbsScaler()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.max_abs_, inc.max_abs_), "max_abs mismatch"


run_test("MaxAbsScaler partial_fit equivalence", test_maxabs_scaler_equivalence)


# ======================================================================
# LabelEncoder
# ======================================================================


def test_label_encoder_partial_fit():
    """New classes appended without changing existing indices."""
    X1 = TimeTensor(
        data=np.zeros((3, 10, 1)),
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        y=np.array(["cat", "dog", "cat"]),
    )
    X2 = TimeTensor(
        data=np.zeros((3, 10, 1)),
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        y=np.array(["fish", "dog", "fish"]),
    )

    enc = LabelEncoder()
    enc.partial_fit(X1)

    cat_idx = enc._class_to_idx["cat"]
    dog_idx = enc._class_to_idx["dog"]

    enc.partial_fit(X2)

    # Existing indices preserved
    assert enc._class_to_idx["cat"] == cat_idx
    assert enc._class_to_idx["dog"] == dog_idx
    # New class added
    assert "fish" in enc._class_to_idx
    assert len(enc.classes_) == 3


run_test("LabelEncoder partial_fit append-only", test_label_encoder_partial_fit)


# ======================================================================
# Interpolate
# ======================================================================


def test_interpolate_partial_fit():
    """Running channel means ≈ full fit means."""
    X = make_dense(B=40)
    X1 = X[:20]
    X2 = X[20:]

    full = Interpolate()
    full.fit(X)

    inc = Interpolate()
    inc.partial_fit(X1)
    inc.partial_fit(X2)

    assert np.allclose(full.channel_means_, inc.channel_means_, atol=1e-10)


run_test("Interpolate partial_fit equivalence", test_interpolate_partial_fit)


# ======================================================================
# Pipeline
# ======================================================================


def test_pipeline_partial_fit():
    """Pipeline.partial_fit chains through steps."""
    X = make_dense(B=20, T=128, C=3)

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("flat", Flatten()),
        ]
    )

    pipe.partial_fit(X)
    assert pipe.is_fitted

    Xt = pipe.transform(X)
    assert Xt.data.ndim == 2
    assert Xt.data.shape[0] == 20


run_test("Pipeline partial_fit", test_pipeline_partial_fit)


def test_pipeline_partial_fit_transform():
    """Pipeline.partial_fit_transform returns valid output."""
    X = make_dense(B=20, T=128, C=3)

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("flat", Flatten()),
        ]
    )

    Xt = pipe.partial_fit_transform(X)
    assert pipe.is_fitted
    assert Xt.data.ndim == 2


run_test("Pipeline partial_fit_transform", test_pipeline_partial_fit_transform)


# ======================================================================
# fit() resets state
# ======================================================================


def test_fit_resets_after_partial_fit():
    """fit() should reset state, not accumulate."""
    X_big = make_dense(B=40, seed=1)
    X_small = make_dense(B=10, seed=2)

    scaler = StandardScaler()
    scaler.partial_fit(X_big)
    scaler.partial_fit(X_big)
    assert scaler.n_samples_seen_ == 40 * 50 * 2

    # fit() resets
    scaler.fit(X_small)
    assert scaler.n_samples_seen_ == 10 * 50

    # Should match fresh fit
    fresh = StandardScaler()
    fresh.fit(X_small)
    assert np.allclose(scaler.mean_, fresh.mean_)
    assert np.allclose(scaler.std_, fresh.std_)


run_test("fit() resets state after partial_fit", test_fit_resets_after_partial_fit)


# ======================================================================
# Stateless transforms
# ======================================================================


def test_stateless_partial_fit():
    """Stateless transforms support partial_fit without error."""
    X = make_dense()
    normalizer = Normalizer()
    normalizer.partial_fit(X)
    assert normalizer.is_fitted
    Xt = normalizer.transform(X)
    assert Xt.data.shape == X.data.shape


run_test("Stateless transform partial_fit", test_stateless_partial_fit)


def test_supports_partial_fit_flag():
    """Check supports_partial_fit is set correctly."""
    assert StandardScaler.supports_partial_fit is True
    assert MinMaxScaler.supports_partial_fit is True
    assert MaxAbsScaler.supports_partial_fit is True
    assert LabelEncoder.supports_partial_fit is True
    assert Normalizer.supports_partial_fit is True
    assert Flatten.supports_partial_fit is True

    from tensorts.transforms.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
    from tensorts.transforms.shape import PCA

    assert RobustScaler.supports_partial_fit is False
    assert PowerTransformer.supports_partial_fit is False
    assert QuantileTransformer.supports_partial_fit is False
    assert PCA.supports_partial_fit is False


run_test("supports_partial_fit flags", test_supports_partial_fit_flag)


# ======================================================================
# SklearnWrapper
# ======================================================================


def test_sklearn_wrapper_partial_fit():
    """SklearnWrapper.partial_fit with SGDClassifier."""
    try:
        from sklearn.linear_model import SGDClassifier
    except ImportError:
        print("  SKIP: sklearn not available")
        return

    from tensorts.compat import SklearnWrapper

    rng = np.random.RandomState(42)
    data = rng.randn(30, 50, 3)
    y = np.array([0, 1, 2] * 10)
    X = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c"],
        y=y,
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("flat", Flatten()),
        ]
    )
    wrapper = SklearnWrapper(pipe, SGDClassifier(random_state=42))

    # First batch — must pass classes= for SGDClassifier
    wrapper.partial_fit(X[:15], classes=np.array([0, 1, 2]))
    # Second batch
    wrapper.partial_fit(X[15:])

    preds = wrapper.predict(X)
    assert preds.shape == (30,)
    assert all(p in [0, 1, 2] for p in preds)


run_test("SklearnWrapper partial_fit", test_sklearn_wrapper_partial_fit)


# ======================================================================
# Multi-batch convergence
# ======================================================================


def test_standard_scaler_many_batches():
    """partial_fit over many small batches ≈ full fit."""
    rng = np.random.RandomState(99)
    data = rng.randn(100, 30, 4) * np.array([1, 5, 10, 50])
    X = TimeTensor(
        data=data,
        dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["a", "b", "c", "d"],
    )

    full = StandardScaler()
    full.fit(X)

    inc = StandardScaler()
    for i in range(0, 100, 5):
        inc.partial_fit(X[i : i + 5])

    assert np.allclose(full.mean_, inc.mean_, atol=1e-10)
    assert np.allclose(full.std_, inc.std_, atol=1e-10)
    assert inc.n_samples_seen_ == 100 * 30


run_test("StandardScaler many small batches", test_standard_scaler_many_batches)


# ======================================================================
# Results
# ======================================================================

total = PASS + FAIL
print("=" * 60)
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
print("=" * 60)
import sys

sys.exit(1 if FAIL > 0 else 0)
