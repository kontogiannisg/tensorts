"""
Tests for FeatureExtractor (single-fn and multi-fn modes) and make_statistical_extractor.

FeatureExtractor operates in two modes:

  Single-fn mode  (fn=...)
    fn(T,) → any shape.  The consumed axis is replaced by output_dims.
    Output can be 2-D (spectrogram), 1-D (resampled), or scalar (energy).
    Works on dense and ragged; preserves windowed leading axes.

  Multi-fn mode  (functions=[...])
    Each fn(T,) → scalar or 1-D.  Results concatenated along new Dim.FEATURE.
    Acts like sklearn FeatureUnion — replaces StatisticalFeatures + FrameFeatureBank.
    Works on dense, ragged, and windowed (B, W, T, C) inputs.

make_statistical_extractor — factory returning a multi-fn FeatureExtractor
  pre-loaded with built-in stats and/or custom callables.

Both dense and ragged inputs are tested for every mode.
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
# PART 1: FeatureExtractor — single-fn mode
# =====================================================================
print("=" * 60)
print("PART 1: FeatureExtractor — single-fn mode")
print("=" * 60)


def test_single_fn_spectrogram_dense():
    """(T,) → (FREQ, T') mel spectrogram — dense (B, T, C) → (B, F, T', C)."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def mel_like(x, n_mels=64, n_fft=512):
        stft = np.abs(np.fft.rfft(
            np.pad(x, (0, max(0, n_fft - len(x)))), n=n_fft
        )) ** 2
        frames = np.array_split(stft, n_mels)
        return np.array([f.sum() for f in frames])  # (n_mels,)

    data = np.random.randn(4, 16000, 1)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    fe = FeatureExtractor(
        fn=mel_like,
        input_dim=Dim.TIME,
        output_dims=(Dim.FREQUENCY,),
        fn_kwargs={"n_mels": 64},
        name="mel_like",
    )
    result = fe.fit_transform(ts)

    assert result.shape == (4, 64, 1)
    assert result.has_dim(Dim.FREQUENCY)
    assert not result.has_dim(Dim.TIME)
    print(f"    mel-like dense: {ts.shape} → {result.shape}")

test("Single-fn dense: (B,T,C) → (B,FREQ,C) fake mel", test_single_fn_spectrogram_dense)


def test_single_fn_stft_stereo():
    """(T,) → (FREQ, T') STFT — stereo: channels preserved."""
    from tensorts import TimeTensor, Dim, STFT

    data = np.random.randn(2, 8000, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["L", "R"])
    result = STFT(n_fft=512, hop_length=256).fit_transform(ts)
    assert result.shape[0] == 2
    assert result.shape[1] == 257        # n_fft//2 + 1
    assert result.n_channels == 2
    print(f"    STFT stereo: {ts.shape} → {result.shape}")

test("Single-fn dense: STFT stereo — both channels preserved", test_single_fn_stft_stereo)


def test_single_fn_resample():
    """(T,) → (T',) resample — same dim type, different length."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def resample_fn(x, target=64):
        indices = np.linspace(0, len(x) - 1, target)
        return np.interp(indices, np.arange(len(x)), x)

    data = np.random.randn(5, 200, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    fe = FeatureExtractor(fn=resample_fn, input_dim=Dim.TIME, output_dims=(Dim.TIME,),
                          fn_kwargs={"target": 64})
    result = fe.fit_transform(ts)
    assert result.shape == (5, 64, 3)
    assert result.has_dim(Dim.TIME)

test("Single-fn dense: resample TIME axis", test_single_fn_resample)


def test_single_fn_scalar_dense():
    """(T,) → scalar: TIME axis removed entirely."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    energy = FeatureExtractor(
        fn=lambda x: np.sum(x ** 2),
        input_dim=Dim.TIME,
        output_dims=(),    # scalar → axis removed
        name="energy",
    )
    data = np.random.randn(6, 100, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    result = energy.fit_transform(ts)
    assert result.shape == (6, 4)
    assert not result.has_dim(Dim.TIME)
    assert result.has_dim(Dim.CHANNEL)

test("Single-fn dense: scalar output (B,T,C) → (B,C)", test_single_fn_scalar_dense)


def test_single_fn_windowed_dense():
    """Windowed (B,W,T,C): single-fn applies per-window, leading dims preserved."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def mel_stub(x, n_mels=32, n_fft=128):
        stft = np.abs(np.fft.rfft(np.pad(x, (0, max(0, n_fft - len(x)))), n=n_fft)) ** 2
        frames = np.array_split(stft, n_mels)
        return np.array([f.sum() for f in frames])

    data = np.random.randn(4, 10, 512, 1)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL))
    fe = FeatureExtractor(fn=mel_stub, input_dim=Dim.TIME,
                          output_dims=(Dim.FREQUENCY,), fn_kwargs={"n_mels": 32})
    result = fe.fit_transform(ts)
    assert result.shape == (4, 10, 32, 1)
    assert result.has_dim(Dim.FREQUENCY)
    assert result.has_dim(Dim.WINDOW)
    print(f"    Windowed mel: {ts.shape} → {result.shape}")

test("Single-fn dense windowed: (B,W,T,C) → (B,W,FREQ,C)", test_single_fn_windowed_dense)


def test_single_fn_scalar_windowed_dense():
    """Windowed scalar (B,W,T,C) → (B,W,C): TIME axis collapsed."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    data = np.random.randn(4, 10, 128, 6)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL))
    rms_fe = FeatureExtractor(fn=lambda x: float(np.sqrt(np.mean(x**2))),
                              input_dim=Dim.TIME, output_dims=())
    result = rms_fe.fit_transform(ts)
    assert result.shape == (4, 10, 6)
    assert not result.has_dim(Dim.TIME)
    print(f"    Windowed scalar: {ts.shape} → {result.shape}")

test("Single-fn dense windowed: (B,W,T,C) scalar → (B,W,C)", test_single_fn_scalar_windowed_dense)


def test_single_fn_ragged_mel():
    """Ragged input: fn applied per sample → dense output."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def mel_stub(x, n_mels=32):
        stft = np.abs(np.fft.rfft(x, n=256)) ** 2
        return np.array([g.sum() for g in np.array_split(stft, n_mels)])

    samples = [np.random.randn(np.random.randint(800, 2000)) for _ in range(5)]
    ts = TimeTensor.from_ragged([s[:, np.newaxis] for s in samples])
    fe = FeatureExtractor(fn=mel_stub, input_dim=Dim.TIME,
                          output_dims=(Dim.FREQUENCY,))
    result = fe.fit_transform(ts)

    assert not result.is_ragged
    assert result.shape == (5, 32, 1)
    print(f"    Ragged mel: {ts.shape} → {result.shape}")

test("Single-fn ragged: variable-length → fixed mel features", test_single_fn_ragged_mel)


def test_single_fn_ragged_scalar():
    """Ragged input scalar fn → dense (B, C)."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    samples = [np.random.randn(np.random.randint(50, 200), 3) for _ in range(6)]
    y = np.array([0, 1, 2, 0, 1, 2])
    ts = TimeTensor.from_ragged(samples, y=y)

    fe = FeatureExtractor(fn=lambda x: np.sum(x**2), input_dim=Dim.TIME, output_dims=())
    result = fe.fit_transform(ts)

    assert result.shape == (6, 3)
    np.testing.assert_array_equal(result.y, y)

test("Single-fn ragged: scalar → dense (B,C), y preserved", test_single_fn_ragged_scalar)


# =====================================================================
# PART 2: FeatureExtractor — multi-fn mode
# =====================================================================
print()
print("=" * 60)
print("PART 2: FeatureExtractor — multi-fn mode")
print("=" * 60)


def test_multi_fn_scalar_dense():
    """Multi-fn with all-scalar fns: (B,T,C) → (B,K,C)."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    data = np.random.randn(6, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["a", "b", "c"])

    fe = FeatureExtractor(
        functions=[
            ("mean",   np.mean),
            ("std",    np.std),
            ("energy", lambda x: float(np.sum(x**2))),
        ],
        input_dim=Dim.TIME,
    )
    result = fe.fit_transform(ts)

    assert result.shape == (6, 3, 3)       # (B, K=3 features, C=3 channels)
    assert result.has_dim(Dim.FEATURE)
    assert not result.has_dim(Dim.TIME)
    assert fe.feature_names_out_ == ["mean", "std", "energy"]

test("Multi-fn dense: all-scalar (B,T,C) → (B,K,C)", test_multi_fn_scalar_dense)


def test_multi_fn_vector_dense():
    """Multi-fn with vector fn: (B,T,C) → (B,K_total,C)."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def top3_fft(x):
        return np.sort(np.abs(np.fft.rfft(x)))[-3:]

    data = np.random.randn(4, 128, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    fe = FeatureExtractor(
        functions=[
            ("mean", np.mean),        # scalar → K=1
            ("top3", top3_fft),       # vector → K=3
        ],
        input_dim=Dim.TIME,
    )
    result = fe.fit_transform(ts)

    assert result.shape == (4, 4, 2)    # K_total = 1 + 3 = 4
    assert fe.feature_names_out_ == ["mean", "top3__0", "top3__1", "top3__2"]

test("Multi-fn dense: mixed scalar + vector → (B,K_total,C)", test_multi_fn_vector_dense)


def test_multi_fn_windowed_dense():
    """Multi-fn on windowed (B,W,T,C) → (B,W,K,C)."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    data = np.random.randn(3, 8, 64, 4)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.WINDOW, Dim.TIME, Dim.CHANNEL))

    fe = FeatureExtractor(
        functions=[("mean", np.mean), ("rms", lambda x: float(np.sqrt(np.mean(x**2))))],
        input_dim=Dim.TIME,
    )
    result = fe.fit_transform(ts)

    assert result.shape == (3, 8, 2, 4)   # BATCH, WINDOW preserved; K=2, C=4
    assert result.has_dim(Dim.WINDOW)
    assert result.has_dim(Dim.FEATURE)
    print(f"    Windowed multi-fn: {ts.shape} → {result.shape}")

test("Multi-fn dense windowed: (B,W,T,C) → (B,W,K,C)", test_multi_fn_windowed_dense)


def test_multi_fn_ragged():
    """Multi-fn on ragged: variable lengths → dense (B,K,C)."""
    from tensorts import TimeTensor, FeatureExtractor, Dim

    samples = [np.random.randn(100 + i * 50, 3) for i in range(5)]
    y = np.array([0, 1, 0, 1, 0])
    ts = TimeTensor.from_ragged(samples, channels=["x", "y", "z"], y=y)

    fe = FeatureExtractor(
        functions=[("mean", np.mean), ("std", np.std), ("min", np.min)],
        input_dim=Dim.TIME,
    )
    result = fe.fit_transform(ts)

    assert not result.is_ragged
    assert result.shape == (5, 3, 3)        # B=5, K=3 fns, C=3 channels
    np.testing.assert_array_equal(result.y, y)

test("Multi-fn ragged: variable lengths → dense (B,K,C)", test_multi_fn_ragged)


def test_multi_fn_per_fn_kwargs():
    """Multi-fn: per-function kwargs via 3-tuples."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    def top_k(x, k=5):
        return np.sort(np.abs(x))[-k:]

    data = np.random.randn(3, 100, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    fe = FeatureExtractor(
        functions=[
            ("mean",  np.mean),
            ("top3",  top_k, {"k": 3}),
            ("top5",  top_k, {"k": 5}),
        ],
        input_dim=Dim.TIME,
    )
    result = fe.fit_transform(ts)

    assert result.shape == (3, 9, 2)     # 1 + 3 + 5 = 9 features
    names = fe.feature_names_out_
    assert "mean" in names
    assert "top3__0" in names
    assert "top5__4" in names

test("Multi-fn: per-function kwargs via 3-tuples", test_multi_fn_per_fn_kwargs)


def test_multi_fn_channels_preserved():
    """Multi-fn: channel names on CHANNEL axis are original channel names."""
    from tensorts import TimeTensor, Dim, FeatureExtractor

    data = np.random.randn(4, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["acc_x", "acc_y", "acc_z"])

    fe = FeatureExtractor(functions=[("mean", np.mean), ("std", np.std)],
                          input_dim=Dim.TIME)
    result = fe.fit_transform(ts)

    assert result.channels.names == ("acc_x", "acc_y", "acc_z")
    assert result.shape[1] == 2      # K=2 features

test("Multi-fn: original channel names preserved on CHANNEL axis", test_multi_fn_channels_preserved)


def test_multi_fn_get_set_params():
    """Multi-fn: get_params / set_params roundtrip (sklearn compat)."""
    from tensorts import FeatureExtractor, Dim
    from sklearn.base import clone

    fe = FeatureExtractor(
        functions=[("mean", np.mean), ("std", np.std)],
        input_dim=Dim.TIME,
    )
    params = fe.get_params()
    assert "functions" in params
    assert "input_dim" in params

    fe2 = clone(fe)
    assert fe2.functions is not None
    assert fe2._mode == "multi"

test("Multi-fn: get_params / set_params / clone roundtrip", test_multi_fn_get_set_params)


def test_multi_fn_errors():
    """Multi-fn: validation errors at construction."""
    from tensorts import FeatureExtractor, Dim

    try:
        FeatureExtractor(fn=np.mean, functions=[("mean", np.mean)], input_dim=Dim.TIME)
        assert False, "should raise"
    except ValueError:
        pass

    try:
        FeatureExtractor(functions=[], input_dim=Dim.TIME)
        assert False, "should raise"
    except ValueError:
        pass

test("Multi-fn: ValueError on invalid construction", test_multi_fn_errors)


# =====================================================================
# PART 3: make_statistical_extractor
# =====================================================================
print()
print("=" * 60)
print("PART 3: make_statistical_extractor")
print("=" * 60)


def test_stat_extractor_default():
    """Default features: mean, std, min, max."""
    from tensorts import TimeTensor, Dim, make_statistical_extractor

    data = np.random.randn(5, 100, 3)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    fe = make_statistical_extractor()
    result = fe.fit_transform(ts)
    assert result.shape == (5, 4, 3)
    assert fe.feature_names_out_ == ["mean", "std", "min", "max"]

test("make_statistical_extractor: default (mean, std, min, max)", test_stat_extractor_default)


def test_stat_extractor_all_builtins():
    """All 13 built-in statistical features."""
    from tensorts import TimeTensor, Dim, make_statistical_extractor

    all_features = ["mean", "std", "var", "min", "max", "median",
                    "rms", "range", "energy", "skew", "kurtosis", "iqr", "zcr"]
    data = np.random.randn(4, 200, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))
    fe = make_statistical_extractor(all_features)
    result = fe.fit_transform(ts)
    assert result.shape == (4, 13, 2)
    assert all(n in fe.feature_names_out_ for n in all_features)

test("make_statistical_extractor: all 13 built-in features", test_stat_extractor_all_builtins)


def test_stat_extractor_ragged():
    """make_statistical_extractor on ragged data."""
    from tensorts import TimeTensor, make_statistical_extractor

    samples = [np.random.randn(np.random.randint(80, 200), 4) for _ in range(10)]
    y = np.random.randint(0, 2, 10)
    ts = TimeTensor.from_ragged(samples, y=y)

    fe = make_statistical_extractor(["mean", "std", "rms"])
    result = fe.fit_transform(ts)

    assert result.shape == (10, 3, 4)
    np.testing.assert_array_equal(result.y, y)

test("make_statistical_extractor: ragged → dense (B,K,C)", test_stat_extractor_ragged)


def test_stat_extractor_custom_callable():
    """Pass a custom callable instead of a string."""
    from tensorts import TimeTensor, Dim, make_statistical_extractor

    def dominant_freq(x):
        freqs = np.fft.rfftfreq(len(x))
        mag = np.abs(np.fft.rfft(x))
        return float(freqs[np.argmax(mag)])

    data = np.random.randn(4, 128, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    fe = make_statistical_extractor([
        "mean",
        ("dominant_freq", dominant_freq),
    ])
    result = fe.fit_transform(ts)
    assert result.shape == (4, 2, 2)
    assert "dominant_freq" in fe.feature_names_out_

test("make_statistical_extractor: custom callable mixed with built-in", test_stat_extractor_custom_callable)


def test_stat_extractor_custom_with_kwargs():
    """Custom callable with per-function kwargs (3-tuple)."""
    from tensorts import TimeTensor, Dim, make_statistical_extractor

    def top_k_energy(x, k=3):
        return np.sort(np.abs(np.fft.rfft(x)))[-k:]

    data = np.random.randn(3, 128, 2)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    fe = make_statistical_extractor([
        "mean",
        ("top5_fft", top_k_energy, {"k": 5}),
    ])
    result = fe.fit_transform(ts)
    assert result.shape == (3, 6, 2)   # 1 (mean) + 5 (top5_fft) = 6
    assert "top5_fft__0" in fe.feature_names_out_
    assert "top5_fft__4" in fe.feature_names_out_

test("make_statistical_extractor: custom callable with kwargs (3-tuple)", test_stat_extractor_custom_with_kwargs)


def test_stat_extractor_unknown_string_raises():
    """Unknown string feature name raises ValueError."""
    from tensorts import make_statistical_extractor

    try:
        make_statistical_extractor(["mean", "invalid_feature"])
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "invalid_feature" in str(e)

test("make_statistical_extractor: unknown string → ValueError", test_stat_extractor_unknown_string_raises)


def test_stat_extractor_bad_entry_raises():
    """Non-string non-tuple entry raises TypeError."""
    from tensorts import make_statistical_extractor

    try:
        make_statistical_extractor([np.mean])   # function without name
        assert False, "should raise TypeError"
    except TypeError:
        pass

test("make_statistical_extractor: bare function without name → TypeError", test_stat_extractor_bad_entry_raises)


# =====================================================================
# PART 4: Real-world pipeline simulations
# =====================================================================
print()
print("=" * 60)
print("PART 4: Real-world pipeline simulations")
print("=" * 60)


def test_audio_mel_pipeline():
    """Audio → PreEmphasis → MelSpectrogram → Log → Scale."""
    from tensorts import (
        TimeTensor, Dim, Pipeline,
        PreEmphasis, MelSpectrogram, LogTransform, StandardScaler,
    )
    sr = 16000
    data = np.random.randn(20, sr, 1)
    y = np.random.randint(0, 2, 20)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["audio"], y=y, metadata={"sampling_rate": sr})

    pipe = Pipeline([
        ("pe",    PreEmphasis(coeff=0.97)),
        ("mel",   MelSpectrogram(n_fft=512, hop_length=160, n_mels=64, sampling_rate=sr)),
        ("log",   LogTransform()),
        ("scale", StandardScaler()),
    ])
    result = pipe.fit_transform(ts)
    assert result.shape[0] == 20
    assert result.shape[1] == 64
    assert result.has_dim(Dim.FREQUENCY)
    np.testing.assert_array_equal(result.y, y)
    print(f"    Audio→Mel: {ts.shape} → {result.shape}")

test("PIPELINE: Audio → PreEmphasis → Mel → Log → Scale", test_audio_mel_pipeline)


def test_har_ragged_statistical_pipeline():
    """HAR ragged → StandardScaler → SlidingWindow → make_statistical_extractor → Flatten."""
    from tensorts import (
        TimeTensor, Pipeline, StandardScaler, SlidingWindow, Flatten,
        make_statistical_extractor,
    )
    rng = np.random.default_rng(1)
    samples = [rng.standard_normal((rng.integers(150, 400), 6)) for _ in range(40)]
    labels = np.tile(np.arange(4), 10)

    ts = TimeTensor.from_ragged(samples,
                                channels=["ax", "ay", "az", "gx", "gy", "gz"],
                                y=labels)

    pipe = Pipeline([
        ("scale",  StandardScaler()),
        ("window", SlidingWindow(size=64, stride=32, label_strategy="repeat", flatten_batch=True)),
        ("stats",  make_statistical_extractor(["mean", "std", "rms", "zcr", "energy"])),
        ("flat",   Flatten()),
    ])
    result = pipe.fit_transform(ts)

    # 5 stats × 6 channels = 30 features
    assert result.shape[-1] == 30
    assert result.y is not None
    print(f"    HAR: {ts.n_samples} ragged → {result.shape}")

test("PIPELINE: HAR ragged → StandardScaler → Window → Stats → Flatten",
     test_har_ragged_statistical_pipeline)


def test_eeg_psd_channel_select():
    """EEG → STFT → ChannelSelect motor cortex → StandardScaler."""
    from tensorts import (
        TimeTensor, Dim, Pipeline, STFT, ChannelSelect, StandardScaler,
    )
    sr = 256
    data = np.random.randn(10, sr * 2, 8)
    ts = TimeTensor(
        data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
        channels=["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
        channel_groups={"motor": ["C3", "C4"]},
    )

    pipe = Pipeline([
        ("psd",   STFT(n_fft=64, hop_length=32)),
        ("motor", ChannelSelect("motor", StandardScaler())),
    ])
    result = pipe.fit_transform(ts)

    assert result.n_channels == 8    # all channels kept
    print(f"    EEG: {ts.shape} → {result.shape}")

test("PIPELINE: EEG → STFT → ChannelSelect (motor) → Scale", test_eeg_psd_channel_select)


def test_multi_fn_in_windowed_ragged_pipeline():
    """Ragged sensor → scale → window → multi-fn extractor → flatten."""
    from tensorts import (
        TimeTensor, Dim, Pipeline, RobustScaler, SlidingWindow, Flatten,
        FeatureExtractor,
    )

    def autocorr_lag1(x):
        if len(x) < 2:
            return np.float64(0.0)
        return np.float64(np.corrcoef(x[:-1], x[1:])[0, 1])

    rng = np.random.default_rng(42)
    samples = [rng.standard_normal((rng.integers(200, 500), 4)) for _ in range(20)]
    y = np.random.randint(0, 3, 20)
    ts = TimeTensor.from_ragged(samples, y=y,
                                channels=["a", "b", "c", "d"])

    fe = FeatureExtractor(
        functions=[
            ("mean",  np.mean),
            ("std",   np.std),
            ("lag1",  autocorr_lag1),
        ],
        input_dim=Dim.TIME,
    )

    pipe = Pipeline([
        ("scale",  RobustScaler()),
        ("window", SlidingWindow(size=50, stride=25, label_strategy="repeat", flatten_batch=True)),
        ("fe",     fe),
        ("flat",   Flatten()),
    ])
    result = pipe.fit_transform(ts)

    # 3 features × 4 channels = 12
    assert result.shape[-1] == 12
    assert result.y is not None
    print(f"    Multi-fn ragged pipeline: {ts.n_samples} samples → {result.shape}")

test("PIPELINE: ragged → RobustScaler → Window → multi-fn FE → Flatten",
     test_multi_fn_in_windowed_ragged_pipeline)


def test_make_stat_extractor_full_pipeline():
    """make_statistical_extractor with mixed built-ins + custom fn in a pipeline."""
    from tensorts import (
        TimeTensor, Dim, Pipeline, StandardScaler, SlidingWindow, Flatten,
        make_statistical_extractor,
    )

    def fft_energy(x):
        return float(np.sum(np.abs(np.fft.rfft(x)) ** 2))

    data = np.random.randn(10, 200, 3)
    y = np.random.randint(0, 2, 10)
    ts = TimeTensor(data=data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL),
                    channels=["x", "y", "z"], y=y)

    fe = make_statistical_extractor([
        "mean", "std",
        ("fft_energy", fft_energy),
    ])

    pipe = Pipeline([
        ("scale",  StandardScaler()),
        ("window", SlidingWindow(size=50, stride=25)),
        ("fe",     fe),
        ("flat",   Flatten()),
    ])
    result = pipe.fit_transform(ts)

    # Dense SlidingWindow → (B, K, W, C). After fe: (B, K, 3, C). After flatten: (B, K*3*3).
    # Verify feature count (3 fns × 3 channels) divides evenly into the flat dimension.
    assert result.shape[0] == 10
    assert result.shape[-1] % 9 == 0    # 3 fns × 3 channels
    print(f"    make_statistical_extractor (mixed): {ts.shape} → {result.shape}")

test("PIPELINE: make_statistical_extractor (built-ins + custom) in pipeline",
     test_make_stat_extractor_full_pipeline)


# =====================================================================
# Flatten descriptive naming
# =====================================================================
print("\n--- Flatten Descriptive Naming ---")

def test_flatten_descriptive_names_after_multi_fn():
    """Flatten after multi-fn FeatureExtractor produces 'fn__channel' names."""
    from tensorts import TimeTensor, Dim, Pipeline
    from tensorts.transforms import Flatten
    from tensorts.transforms.functional import FeatureExtractor

    data = np.random.randn(5, 100, 3)
    channels = ["acc_x", "acc_y", "acc_z"]
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=channels)

    fe = FeatureExtractor(
        functions=[("mean", np.mean), ("std", np.std)],
        input_dim=Dim.TIME,
    )
    fe.fit(ts)
    extracted = fe.transform(ts)

    flat = Flatten()
    flat.fit(extracted)
    result = flat.transform(extracted)

    assert result.shape == (5, 6), f"Expected (5, 6), got {result.shape}"
    expected_names = (
        "mean__acc_x", "mean__acc_y", "mean__acc_z",
        "std__acc_x", "std__acc_y", "std__acc_z",
    )
    assert tuple(result.channels.names) == expected_names, (
        f"Expected {expected_names}, got {result.channels.names}"
    )
    print(f"    channels after flatten: {result.channels.names}")

test("Flatten produces descriptive names after multi-fn FeatureExtractor",
     test_flatten_descriptive_names_after_multi_fn)


def test_flatten_generic_names_without_feature_metadata():
    """Flatten without feature metadata falls back to f_0, f_1, ..."""
    from tensorts import TimeTensor, Dim
    from tensorts.transforms import Flatten

    data = np.random.randn(5, 10, 3)
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL))

    flat = Flatten()
    flat.fit(ts)
    result = flat.transform(ts)

    assert result.shape == (5, 30)
    expected = tuple(f"f_{i}" for i in range(30))
    assert tuple(result.channels.names) == expected, (
        f"Expected generic names, got {result.channels.names}"
    )
    print(f"    channels after flatten (no metadata): {result.channels.names[:5]}...")

test("Flatten falls back to generic names without feature metadata",
     test_flatten_generic_names_without_feature_metadata)


def test_flatten_descriptive_names_in_pipeline():
    """Pipeline: FeatureExtractor → Flatten produces descriptive names."""
    from tensorts import TimeTensor, Dim, Pipeline
    from tensorts.transforms import Flatten
    from tensorts.transforms.features import make_statistical_extractor

    data = np.random.randn(8, 50, 2)
    channels = ["temp", "humidity"]
    ts = TimeTensor(data, dims=(Dim.BATCH, Dim.TIME, Dim.CHANNEL), channels=channels)

    pipe = Pipeline([
        ("stats", make_statistical_extractor(["mean", "std", "min"])),
        ("flatten", Flatten()),
    ])
    result = pipe.fit_transform(ts)

    assert result.shape == (8, 6)
    expected = (
        "mean__temp", "mean__humidity",
        "std__temp", "std__humidity",
        "min__temp", "min__humidity",
    )
    assert tuple(result.channels.names) == expected, (
        f"Expected {expected}, got {result.channels.names}"
    )
    print(f"    pipeline channels: {result.channels.names}")

test("Pipeline: FeatureExtractor → Flatten descriptive names",
     test_flatten_descriptive_names_in_pipeline)


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
