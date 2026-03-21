"""
Challenging sklearn compatibility tests for tensorts.

All problems start from ragged (B, Ti, C) input and exercise:
- FunctionTransform and FeatureExtractor
- clone(), cross_val_score, cross_validate, GridSearchCV
- Label propagation through SlidingWindow
- Nested get_params / set_params through deep pipelines

Run with the 'aiot' conda env which has sklearn installed:
  /opt/homebrew/Caskroom/miniconda/base/envs/aiot/bin/python test_sklearn_compat_problems.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)

from tensorts import (
    Dim,
    Flatten,
    Pipeline,
    RobustScaler,
    SklearnWrapper,
    SlidingWindow,
    StandardScaler,
    TimeTensor,
    make_statistical_extractor,
)
from tensorts.transforms.functional import FeatureExtractor, FunctionTransform, StatefulFunctionTransform
from tensorts.transforms.base import Scope


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generators
# ──────────────────────────────────────────────────────────────────────────────

def make_har_data(n_subjects=12, n_classes=3, channels=6, seed=42):
    """
    HAR-style dataset: each subject has one variable-length recording per
    class (activity). Recording lengths vary by subject (500-1500 samples).
    Returns ragged TimeTensor with shape (B, Ti, 6).
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for s in range(n_subjects):
        for c in range(n_classes):
            T = rng.integers(500, 1500)
            # Class-specific signal: sinusoid at different frequencies
            t = np.linspace(0, 1, T)
            base = np.sin(2 * np.pi * (c + 1) * 3 * t)
            signal = np.stack([base + 0.3 * rng.standard_normal(T)
                               for _ in range(channels)], axis=-1)
            samples.append(signal.astype(np.float32))
            labels.append(c)

    return TimeTensor.from_ragged(
        samples,
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        y=np.array(labels),
    )


def make_audio_data(n_clips=24, channels=2, seed=7):
    """
    Audio-like dataset: variable-length stereo clips (1000-4000 samples).
    Binary classification (speech vs. noise).
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for i in range(n_clips):
        T = rng.integers(1000, 4000)
        label = i % 2
        if label == 0:  # speech-like: harmonic
            t = np.linspace(0, 1, T)
            sig = np.column_stack([
                np.sin(2 * np.pi * 200 * t) + 0.1 * rng.standard_normal(T),
                np.sin(2 * np.pi * 400 * t) + 0.1 * rng.standard_normal(T),
            ])
        else:  # noise
            sig = rng.standard_normal((T, 2))
        samples.append(sig.astype(np.float32))
        labels.append(label)

    return TimeTensor.from_ragged(samples, channels=["left", "right"],
                                  y=np.array(labels))


def make_biosignal_data(n_recordings=20, channels=4, seed=13):
    """
    ECG/biosignal-style: variable-length recordings (800-2500 samples),
    4 channels with very different scales (mV, bpm, RR-interval, SpO2).
    Binary classification.
    Includes mild NaN contamination in channel 3.
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for i in range(n_recordings):
        T = rng.integers(800, 2500)
        label = i % 2
        t = np.linspace(0, 1, T)
        # ch0: ECG-like mV (~1.0 scale)
        # ch1: heart rate bpm (~70 scale)
        # ch2: RR interval ms (~800 scale)
        # ch3: SpO2 % (~98 scale) with some NaNs
        ch0 = np.sin(2 * np.pi * 1.2 * t * (1 + 0.2 * label)) + 0.05 * rng.standard_normal(T)
        ch1 = 70 + 5 * label + rng.standard_normal(T)
        ch2 = 800 - 30 * label + 10 * rng.standard_normal(T)
        ch3 = 98 + rng.standard_normal(T) * 0.5
        # inject ~2% NaNs in ch3
        nan_idx = rng.choice(T, size=T // 50, replace=False)
        ch3[nan_idx] = np.nan
        sig = np.column_stack([ch0, ch1, ch2, ch3]).astype(np.float64)
        samples.append(sig)
        labels.append(label)

    return TimeTensor.from_ragged(
        samples,
        channels=["ecg", "hr", "rr", "spo2"],
        y=np.array(labels),
    )


def make_financial_data(n_assets=6, n_periods=20, channels=3, seed=99):
    """
    Financial-style: n_periods variable-length price series per asset group.
    3 channels: log-price, volume, volatility (very different scales).
    Regression target: next-period return.
    """
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for a in range(n_assets):
        for p in range(n_periods):
            T = rng.integers(300, 1000)
            # log-price: random walk
            log_price = np.cumsum(rng.standard_normal(T) * 0.01)
            # volume: log-normal, much larger scale
            volume = np.exp(rng.standard_normal(T) + 8)
            # volatility: GARCH-like, small values
            vol = np.abs(rng.standard_normal(T)) * 0.02
            sig = np.column_stack([log_price, volume, vol]).astype(np.float64)
            samples.append(sig)
            # Regression target: total return over period
            labels.append(float(log_price[-1] - log_price[0]))

    return TimeTensor.from_ragged(samples, channels=["log_price", "volume", "vol"],
                                  y=np.array(labels))


# ──────────────────────────────────────────────────────────────────────────────
# Problem 1: HAR Classification
#
# Challenge: ragged (B, Ti, 6) →
#   FunctionTransform(pre-emphasis, PER_CHANNEL) — preserves ragged
#   StandardScaler — preserves ragged
#   SlidingWindow(size=100, stride=50) — ragged → dense, multiplies samples, updates labels
#   StatisticalFeatures — collapses TIME axis
#   Flatten — → (B, F)
#   PCA(n_components=0.95) — dimensionality reduction
#
# Tricky parts:
#   - clone() must survive FunctionTransform (fn is a lambda)
#   - SlidingWindow multiplies B by K, labels must propagate correctly
#   - StratifiedKFold works on the expanded label vector from Xt.y
#   - get_params / set_params must work through the full depth
# ──────────────────────────────────────────────────────────────────────────────

def problem1_har_crossval():
    print("\n" + "="*70)
    print("PROBLEM 1: HAR — ragged → FunctionTransform → SlidingWindow → cross_val_score")
    print("="*70)

    X = make_har_data()
    y = X.y
    print(f"  Input: {X}  |  classes: {np.unique(y)}")

    # Pre-emphasis: x[t] - 0.97 * x[t-1], applied per channel independently
    def pre_emphasis(signal_1d: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        out = np.empty_like(signal_1d)
        out[0] = signal_1d[0]
        out[1:] = signal_1d[1:] - coeff * signal_1d[:-1]
        return out

    from tensorts.transforms.shape import PCA

    pipe = Pipeline([
        ("pre_emph", FunctionTransform(pre_emphasis, scope=Scope.PER_CHANNEL,
                                       fn_kwargs={"coeff": 0.97}, name="pre_emphasis")),
        ("scaler",   StandardScaler()),
        ("window",   SlidingWindow(size=100, stride=50, label_strategy="repeat", flatten_batch=True)),
        ("stats",    make_statistical_extractor(["mean", "std", "min", "max", "energy"])),
        ("flatten",  Flatten()),
        ("pca",      PCA(n_components=0.95)),
    ])

    # ── 1a. Basic fit_transform
    Xt = pipe.fit_transform(X, y)
    print(f"  After pipeline: {Xt}  y_shape={Xt.y.shape if Xt.y is not None else None}")
    assert Xt.y is not None, "Labels must survive SlidingWindow"
    assert Xt.n_samples > X.n_samples, "SlidingWindow should expand sample count"

    # ── 1b. clone() must survive FunctionTransform with lambda-like fn
    cloned_pipe = clone(pipe)
    print(f"  clone() ok — pre_emph fn preserved: {type(cloned_pipe['pre_emph'].fn)}")
    assert callable(cloned_pipe["pre_emph"].fn)

    # ── 1c. get_params / set_params roundtrip
    params = pipe.get_params(deep=True)
    nested_keys = [k for k in params if "__" in k]
    print(f"  Pipeline nested param keys: {nested_keys}")
    assert "pre_emph__fn_kwargs" in params, "fn_kwargs must appear in get_params"
    assert "pca__n_components" in params

    pipe.set_params(**{"scaler__eps": 1e-7, "pca__n_components": 0.90})
    assert pipe["scaler"].eps == 1e-7
    assert pipe["pca"].n_components == 0.90
    print("  set_params roundtrip ok")

    # ── 1d. cross_val_score — NOTE: y must come from Xt.y (expanded by SlidingWindow)
    #    The wrapper's fit() propagates Xt.y, so we pass the ORIGINAL y here and
    #    let the pipeline handle the expansion. But cross_val_score also slices y
    #    externally before calling fit(). This is the key tension:
    #    sklearn slices y[train_idx], but the real label count changes after windowing.
    #
    #    Expected behaviour: wrapper.fit(X_train, y_train) internally calls
    #    pipeline.fit_transform(X_train, y_train) → Xt_train with expanded Xt.y
    #    Then fits estimator on Xt_train.data, Xt_train.y  ← uses propagated labels
    #
    #    wrapper.score(X_test, y_test) → pipeline.transform(X_test) → Xt_test
    #    estimator.score(Xt_test.data, y_test)  ← but y_test length != Xt_test.n_samples!
    #    This WILL fail unless score uses Xt.y instead.
    print(f"X shape {X.shape}, y shape {y.shape}")
    wrapper = SklearnWrapper(pipe, LogisticRegression(max_iter=300))
    try:
        scores = cross_val_score(wrapper, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0))
        print(f"  cross_val_score: {scores.round(3)}  mean={scores.mean():.3f}")
    except Exception as e:
        print(f"  [EXPECTED FAILURE] cross_val_score failed: {type(e).__name__}: {e}")
        print("  → ROOT CAUSE: score(X_test, y_test) uses sklearn-sliced y_test (length=B_test)")
        print("    but Xt_test has B_test * K windows. score() must use Xt.y instead.")
        print("  → NEXT STEP: SklearnWrapper.score() should prefer Xt.y when lengths differ.")

    print("  Problem 1 done.")


# ──────────────────────────────────────────────────────────────────────────────
# Problem 2: Audio — FeatureExtractor + GridSearchCV
#
# Challenge: ragged (B, Ti, 2) →
#   StandardScaler — per-channel, over all samples
#   FeatureExtractor(fft_energy_per_band, TIME → FREQUENCY) — changes shape
#   Flatten — → (B, F)
#   GridSearchCV over pipeline__pca__n_components and estimator__C
#
# Tricky parts:
#   - FeatureExtractor changes (B, T, C) → (B, n_bands, C), then Flatten → (B, n_bands*C)
#   - clone() must deepcopy the fn inside FeatureExtractor
#   - GridSearchCV calls clone + set_params on the full wrapper
#   - FeatureExtractor output_dims must be in get_params (was a bug: output_dims_spec)
# ──────────────────────────────────────────────────────────────────────────────

def problem2_audio_gridsearch():
    print("\n" + "="*70)
    print("PROBLEM 2: Audio — FeatureExtractor(FFT bands) + GridSearchCV")
    print("="*70)

    X = make_audio_data()
    y = X.y
    print(f"  Input: {X}  |  classes: {np.unique(y)}")

    N_BANDS = 8

    def fft_energy_bands(signal_1d: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
        """Compute energy in n_bands frequency bands. Returns (n_bands,)."""
        spec = np.abs(np.fft.rfft(signal_1d)) ** 2
        band_size = max(1, len(spec) // n_bands)
        bands = np.array([spec[i * band_size:(i + 1) * band_size].mean()
                          for i in range(n_bands)])
        return np.log1p(bands)

    from tensorts.transforms.shape import PCA

    pipe = Pipeline([
        ("scaler",   StandardScaler()),
        ("fft_feat", FeatureExtractor(
            fn=fft_energy_bands,
            input_dim=Dim.TIME,
            output_dims=(Dim.FREQUENCY,),
            fn_kwargs={"n_bands": N_BANDS},
            name="fft_energy_bands",
        )),
        ("flatten",  Flatten()),
        ("pca",      PCA(n_components=10)),
    ])

    # ── 2a. Fit-transform sanity
    Xt = pipe.fit_transform(X, y)
    print(f"  After pipeline: {Xt}")
    assert Xt.dims == (Dim.BATCH, Dim.FEATURE), f"Expected (BATCH, FEATURE), got {Xt.dims}"

    # ── 2b. clone() — FeatureExtractor.get_params must return actual fn and Dim objects
    cloned = clone(pipe)
    assert callable(cloned["fft_feat"].fn), "fn must survive clone"
    assert cloned["fft_feat"].output_dims == (Dim.FREQUENCY,), \
        f"output_dims must survive clone, got {cloned['fft_feat'].output_dims}"
    assert cloned["fft_feat"].input_dim == Dim.TIME
    print(f"  clone() ok — FeatureExtractor fn={cloned['fft_feat'].fn.__name__}, "
          f"output_dims={cloned['fft_feat'].output_dims}")

    # ── 2c. get_params deep inspection
    params = pipe.get_params(deep=True)
    print(f"  fft_feat params: { {k: v for k, v in params.items() if 'fft_feat' in k} }")
    assert "fft_feat__input_dim" in params
    assert "fft_feat__output_dims" in params
    assert "fft_feat__fn_kwargs" in params

    # ── 2d. GridSearchCV — exercises clone + set_params end-to-end
    wrapper = SklearnWrapper(pipe, LogisticRegression(max_iter=500))
    param_grid = {
        "pipeline__pca__n_components": [4, 8],
        "estimator__C": [0.1, 1.0],
    }
    gs = GridSearchCV(wrapper, param_grid, cv=3, refit=True, n_jobs=1)
    try:
        gs.fit(X, y)
        print(f"  GridSearchCV best params: {gs.best_params_}")
        print(f"  GridSearchCV best score:  {gs.best_score_:.3f}")
    except Exception as e:
        print(f"  [FAILURE] GridSearchCV: {type(e).__name__}: {e}")
        raise

    print("  Problem 2 done.")

# ──────────────────────────────────────────────────────────────────────────────
# Problem 3: Biosignal — StatefulFunctionTransform + heterogeneous channels
#
# Challenge: ragged (B, Ti, 4) with NaNs in channel 3 →
#   Interpolate (fill NaNs)
#   RobustScaler (handles different channel scales, ragged)
#   StatefulFunctionTransform(per-channel peak normalization) — learns per-channel max
#   FunctionTransform(first-difference derivative, PER_CHANNEL) — changes length T→T-1
#   FeatureExtractor(scalar: total energy per channel, TIME→scalar) — collapses TIME
#   Flatten (→ B, C) — already (B, C) after scalar FeatureExtractor
#   cross_validate (return train scores)
#
# Tricky parts:
#   - StatefulFunctionTransform clone must preserve fit_fn and transform_fn
#   - FunctionTransform changes T to T-1: ragged stays ragged
#   - FeatureExtractor with scalar output_dims=() collapses TIME
#   - After scalar extraction ragged → dense: each sample becomes (C,)
# ──────────────────────────────────────────────────────────────────────────────

def problem3_biosignal_stateful():
    print("\n" + "="*70)
    print("PROBLEM 3: Biosignal — StatefulFunctionTransform + FeatureExtractor(scalar)")
    print("="*70)

    from tensorts import Interpolate

    X = make_biosignal_data()
    y = X.y
    print(f"  Input: {X}  |  classes: {np.unique(y)}")

    # Peak normalization: fit = learn max per channel, transform = divide by max
    def fit_peak(channel_1d: np.ndarray) -> float:
        return float(np.nanmax(np.abs(channel_1d))) + 1e-8

    def transform_peak(channel_1d: np.ndarray, peak: float) -> np.ndarray:
        return channel_1d / peak

    # First difference: x[t] - x[t-1]  (T → T-1)
    def first_diff(signal_1d: np.ndarray) -> np.ndarray:
        return np.diff(signal_1d)

    # Total signal energy: scalar
    def total_energy(signal_1d: np.ndarray) -> np.ndarray:
        return np.array(np.sum(signal_1d ** 2))

    pipe = Pipeline([
        ("interp",  Interpolate(method="linear")),
        ("scaler",  RobustScaler()),
        ("peak",    StatefulFunctionTransform(
            fit_fn=fit_peak, transform_fn=transform_peak,
            scope=Scope.PER_CHANNEL, name="peak_norm",
        )),
        ("diff",    FunctionTransform(first_diff, scope=Scope.PER_CHANNEL, name="first_diff")),
        ("energy",  FeatureExtractor(
            fn=total_energy,
            input_dim=Dim.TIME,
            output_dims=(),          # scalar per channel → collapses TIME entirely
            name="total_energy",
        )),
        ("flatten", Flatten()),
    ])

    # ── 3a. Fit-transform
    Xt = pipe.fit_transform(X, y)
    print(f"  After pipeline: {Xt}  shape={Xt.shape}")
    # Energy is scalar per channel, so output should be (B, C) → after Flatten (B, C)
    assert Xt.n_samples == X.n_samples, "Sample count must be preserved"
    assert Xt.data.ndim == 2, f"Expected 2D (B, F), got {Xt.data.shape}"

    # ── 3b. clone() — StatefulFunctionTransform must preserve fit_fn and transform_fn
    cloned = clone(pipe)
    assert callable(cloned["peak"].fit_fn), "fit_fn must survive clone"
    assert callable(cloned["peak"].transform_fn), "transform_fn must survive clone"
    assert callable(cloned["diff"].fn), "FunctionTransform fn must survive clone"
    assert cloned["energy"].output_dims == (), \
        f"scalar output_dims must survive clone, got {cloned['energy'].output_dims}"
    print(f"  clone() ok — StatefulFunctionTransform and FeatureExtractor preserved")

    # ── 3c. get_params deep inspection
    params = pipe.get_params(deep=True)
    peak_params = {k: v for k, v in params.items() if "peak" in k}
    print(f"  StatefulFunctionTransform params: {list(peak_params.keys())}")
    assert "peak__fit_fn" in params, "fit_fn must appear in get_params"
    assert "peak__transform_fn" in params

    # ── 3d. cross_validate
    wrapper = SklearnWrapper(pipe, LogisticRegression(max_iter=500))
    try:
        cv_results = cross_validate(wrapper, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                    return_train_score=True)
        print(f"  cross_validate test scores:  {cv_results['test_score'].round(3)}")
        print(f"  cross_validate train scores: {cv_results['train_score'].round(3)}")
    except Exception as e:
        print(f"  [FAILURE] cross_validate: {type(e).__name__}: {e}")
        raise

    print("  Problem 3 done.")


# ──────────────────────────────────────────────────────────────────────────────
# Problem 4: Financial Regression — deeply nested GridSearchCV
#
# Challenge: ragged (B, Ti, 3) →
#   FunctionTransform(log1p on volume, PER_CHANNEL via ChannelSelect) — channel-selective
#   RobustScaler — ragged, different scales
#   SlidingWindow(size=50, stride=25) — ragged→dense, regression target repeats
#   FeatureExtractor(autocorrelation lag-1, TIME→scalar) — per window, scalar
#   StatisticalFeatures(["mean","std","min","max"]) — on windowed+extracted features
#   Flatten
#   GridSearchCV over window size AND PCA components AND estimator alpha
#
# Tricky parts:
#   - GridSearchCV must clone and set_params on SlidingWindow.size (int) — nested deep
#   - FeatureExtractor.fn_kwargs must survive clone (carries lag parameter)
#   - Regression: target is continuous, use Ridge + scoring='r2'
#   - SlidingWindow for regression with label_strategy='repeat': y expands too
#   - score(X_test, y_test) label-length mismatch again (same issue as Problem 1)
# ──────────────────────────────────────────────────────────────────────────────

def problem4_financial_gridsearch():
    print("\n" + "="*70)
    print("PROBLEM 4: Financial regression — SlidingWindow + FeatureExtractor + GridSearchCV")
    print("="*70)

    from tensorts.transforms.base import ChannelSelect

    X = make_financial_data()
    y = X.y
    print(f"  Input: {X}  |  y dtype={y.dtype}, y[:5]={y[:5].round(4)}")

    def log1p_fn(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.abs(x)) * np.sign(x)

    def autocorr_lag1(signal_1d: np.ndarray) -> np.ndarray:
        """Lag-1 autocorrelation of a 1D signal. Returns scalar."""
        if len(signal_1d) < 2:
            return np.array(0.0)
        x = signal_1d - signal_1d.mean()
        denom = np.dot(x, x)
        if denom < 1e-10:
            return np.array(0.0)
        return np.array(np.dot(x[:-1], x[1:]) / denom)

    # NOTE: FeatureExtractor with output_dims=() collapses TIME entirely → (B*K, C).
    # StatisticalFeatures CANNOT follow because there is no TIME axis left.
    # The features are the autocorrelation values themselves — just Flatten to (B*K, C).
    # Gap documented: StatisticalFeatures should raise a clear error when TIME is absent.
    pipe = Pipeline([
        ("log_vol",  ChannelSelect(
            channels="volume",
            transform=FunctionTransform(log1p_fn, scope=Scope.PER_CHANNEL, name="log1p"),
        )),
        ("scaler",   RobustScaler()),
        ("window",   SlidingWindow(size=50, stride=25, label_strategy="repeat", flatten_batch=True)),
        ("autocorr", FeatureExtractor(
            fn=autocorr_lag1,
            input_dim=Dim.TIME,
            output_dims=(),   # scalar per channel: (B*K, W, C) → (B*K, C)
            name="autocorr_lag1",
        )),
        ("flatten",  Flatten()),   # (B*K, C) → (B*K, C) — already flat, no-op effectively
    ])

    # ── 4a. Fit-transform sanity
    Xt = pipe.fit_transform(X, y)
    print(f"  After pipeline: {Xt}  shape={Xt.shape}")
    assert Xt.data.ndim == 2

    # ── 4b. Deep clone test — SlidingWindow, ChannelSelect, FeatureExtractor
    cloned = clone(pipe)
    assert cloned["window"].size == 50
    assert cloned["window"].stride == 25
    assert callable(cloned["log_vol"].inner.fn)
    assert callable(cloned["autocorr"].fn)
    print(f"  clone() ok — SlidingWindow(size={cloned['window'].size}), "
          f"FeatureExtractor.fn={cloned['autocorr'].fn.__name__}")

    # ── 4c. Deeply nested get_params / set_params
    params = pipe.get_params(deep=True)
    print(f"  SlidingWindow params in Pipeline: "
          f"{ {k: v for k, v in params.items() if 'window' in k and '__' in k} }")
    assert "window__size" in params
    assert "window__stride" in params

    pipe.set_params(**{"window__size": 60, "window__stride": 30})
    assert pipe["window"].size == 60
    pipe.set_params(**{"window__size": 50, "window__stride": 25})  # restore
    print("  Nested set_params on SlidingWindow ok")

    # ── 4d. GridSearchCV — exercises clone + nested set_params on int params
    wrapper = SklearnWrapper(pipe, Ridge())
    param_grid = {
        "pipeline__window__size":   [30, 50],
        "pipeline__window__stride": [15, 25],
        "estimator__alpha":          [0.1, 1.0],
    }
    gs = GridSearchCV(wrapper, param_grid, cv=3, refit=True, scoring="r2", n_jobs=1)
    try:
        gs.fit(X, y)
        print(f"  GridSearchCV best params: {gs.best_params_}")
        print(f"  GridSearchCV best R²:     {gs.best_score_:.3f}")
    except Exception as e:
        print(f"  [EXPECTED or FAILURE] GridSearchCV: {type(e).__name__}: {e}")
        if "operands could not be broadcast" in str(e) or "y" in str(e).lower():
            print("  → ROOT CAUSE: Same label-length mismatch as Problem 1.")
            print("    score(X_test, y_test) — y_test has original length,")
            print("    but Xt_test has K*B_test windows after SlidingWindow.")
            print("  → NEXT STEP: SklearnWrapper.score() must use Xt.y when available.")
        else:
            raise

    print("  Problem 4 done.")


if __name__ == "__main__":
    problem1_har_crossval()
    problem2_audio_gridsearch()
    problem3_biosignal_stateful()
    problem4_financial_gridsearch()