"""
Feature extraction transforms.

Includes:
- StatisticalFeatures: per-window/per-sample statistical summaries
- RollingFeatures: rolling window statistics
- FrameFeature implementations: ZCR, RMS, SpectralEntropy, SpectralCentroid,
  SpectralRolloff, MFCC, SpectralFlatness
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import FrameFeature, Scope, Transform


# ======================================================================
# StatisticalFeatures
# ======================================================================

class StatisticalFeatures(Transform):
    """Extract statistical features along the time axis.

    Input:  (..., T, C)
    Output: (..., K*C)  where K = number of features, or (..., K) per channel

    The time axis is reduced; each channel produces K scalar features.

    Parameters
    ----------
    features : list of str
        Feature names from: ``'mean'``, ``'std'``, ``'var'``, ``'min'``,
        ``'max'``, ``'median'``, ``'skew'``, ``'kurtosis'``, ``'rms'``,
        ``'iqr'``, ``'range'``, ``'energy'``, ``'zcr'``.
    flatten : bool
        If True, concatenate features across channels into a single FEATURE axis.
        If False, output has shape (..., K, C).
    """

    scope = Scope.PER_CHANNEL

    _FEATURE_FNS = {
        "mean": lambda x, ax: np.mean(x, axis=ax),
        "std": lambda x, ax: np.std(x, axis=ax),
        "var": lambda x, ax: np.var(x, axis=ax),
        "min": lambda x, ax: np.min(x, axis=ax),
        "max": lambda x, ax: np.max(x, axis=ax),
        "median": lambda x, ax: np.median(x, axis=ax),
        "rms": lambda x, ax: np.sqrt(np.mean(x**2, axis=ax)),
        "range": lambda x, ax: np.ptp(x, axis=ax),
        "energy": lambda x, ax: np.sum(x**2, axis=ax),
    }

    def __init__(
        self,
        features: list[str] | None = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.feature_names = features or ["mean", "std", "min", "max"]
        self.flatten = flatten
        self._validate_features()

    def _validate_features(self) -> None:
        valid = set(self._FEATURE_FNS.keys()) | {"skew", "kurtosis", "iqr", "zcr"}
        for f in self.feature_names:
            if f not in valid:
                raise ValueError(
                    f"Unknown feature: {f!r}. Valid: {sorted(valid)}"
                )

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "StatisticalFeatures":
        self._is_fitted = True
        return self

    def _compute_feature(self, data: np.ndarray, name: str, t_ax: int) -> np.ndarray:
        if name in self._FEATURE_FNS:
            return self._FEATURE_FNS[name](data, t_ax)

        if name == "skew":
            m = np.mean(data, axis=t_ax, keepdims=True)
            s = np.std(data, axis=t_ax, keepdims=True) + 1e-8
            return np.mean(((data - m) / s) ** 3, axis=t_ax)

        if name == "kurtosis":
            m = np.mean(data, axis=t_ax, keepdims=True)
            s = np.std(data, axis=t_ax, keepdims=True) + 1e-8
            return np.mean(((data - m) / s) ** 4, axis=t_ax) - 3.0

        if name == "iqr":
            q75 = np.percentile(data, 75, axis=t_ax)
            q25 = np.percentile(data, 25, axis=t_ax)
            return q75 - q25

        if name == "zcr":
            slices_curr = [slice(None)] * data.ndim
            slices_prev = [slice(None)] * data.ndim
            slices_curr[t_ax] = slice(1, None)
            slices_prev[t_ax] = slice(None, -1)
            crossings = np.sign(data[tuple(slices_curr)]) != np.sign(data[tuple(slices_prev)])
            return np.mean(crossings.astype(np.float64), axis=t_ax)

        raise ValueError(f"Unknown feature: {name}")

    def transform(self, X: TimeTensor) -> TimeTensor:
        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else -2

        feats = []
        for name in self.feature_names:
            f = self._compute_feature(data, name, t_ax)
            feats.append(np.expand_dims(f, axis=t_ax))  # keep dim for stacking

        stacked = np.concatenate(feats, axis=t_ax)  # (..., K, C) or similar

        if self.flatten:
            # Flatten the feature and channel axes into one FEATURE axis
            ch_dim = X._find_channel_dim(X.schema)
            ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1
            # After reducing TIME → K features, we have (..., K, C)
            # Flatten K and C: reshape last two dims into one
            shape = list(stacked.shape)
            # K is at t_ax, C is at ch_ax
            leading = shape[:min(t_ax, ch_ax)]
            trailing = shape[max(t_ax, ch_ax) + 1:]
            k_size = shape[t_ax]
            c_size = shape[ch_ax] if ch_ax < len(shape) else 1

            # Reorder to put K, C at the end
            axes_order = [i for i in range(stacked.ndim) if i not in (t_ax, ch_ax)]
            axes_order.extend([t_ax, ch_ax])
            reordered = np.transpose(stacked, axes_order)
            new_shape = list(reordered.shape[:-2]) + [k_size * c_size]
            flat = reordered.reshape(new_shape)

            # Build channel names: feat_chname for each combination
            ch_names = X.channels.names if X.channels else [f"ch_{i}" for i in range(c_size)]
            feat_channel_names = [
                f"{fname}_{cname}" for fname in self.feature_names for cname in ch_names
            ]

            # Build dims: everything except TIME and CHANNEL → FEATURE
            new_dims = [d for d in X.dims if d not in (Dim.TIME, Dim.CHANNEL, Dim.FEATURE)]
            new_dims.append(Dim.FEATURE)

            return X.with_data(
                flat,
                dims=tuple(new_dims),
                channels=ChannelIndex(feat_channel_names),
            )
        else:
            # Replace TIME with FEATURE, keep CHANNEL
            new_dims = list(X.dims)
            t_idx = new_dims.index(Dim.TIME)
            new_dims[t_idx] = Dim.FEATURE
            return X.with_data(stacked, dims=tuple(new_dims))

    def get_params(self) -> dict[str, Any]:
        return {"features": self.feature_names, "flatten": self.flatten}


# ======================================================================
# RollingFeatures
# ======================================================================

class RollingFeatures(Transform):
    """Compute rolling-window statistics along the time axis.

    Output has the same shape as input — each timestep gets the feature
    value computed from its surrounding window.

    Parameters
    ----------
    window_size : int
    features : list of str
        Same vocabulary as StatisticalFeatures.
    center : bool
        If True, the window is centered on each timestep.
    """

    scope = Scope.PER_CHANNEL

    def __init__(
        self,
        window_size: int,
        features: list[str] | None = None,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.feature_names = features or ["mean"]
        self.center = center

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "RollingFeatures":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        data = X.data
        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        T = data.shape[t_ax]

        results = []
        for name in self.feature_names:
            result = np.empty_like(data)
            for t in range(T):
                if self.center:
                    half = self.window_size // 2
                    start = max(0, t - half)
                    end = min(T, t + half + 1)
                else:
                    start = max(0, t - self.window_size + 1)
                    end = t + 1

                slices = [slice(None)] * data.ndim
                slices[t_ax] = slice(start, end)
                window = data[tuple(slices)]

                out_slices = [slice(None)] * data.ndim
                out_slices[t_ax] = t

                if name == "mean":
                    result[tuple(out_slices)] = np.mean(window, axis=t_ax)
                elif name == "std":
                    result[tuple(out_slices)] = np.std(window, axis=t_ax)
                elif name == "min":
                    result[tuple(out_slices)] = np.min(window, axis=t_ax)
                elif name == "max":
                    result[tuple(out_slices)] = np.max(window, axis=t_ax)
                elif name == "median":
                    result[tuple(out_slices)] = np.median(window, axis=t_ax)
                else:
                    raise ValueError(f"RollingFeatures does not support {name!r}.")

            results.append(result)

        if len(results) == 1:
            return X.with_data(results[0])

        # Multiple features: concatenate along channel axis
        stacked = np.concatenate(results, axis=-1)
        if X.channels:
            ch_names = []
            for fname in self.feature_names:
                for cn in X.channels.names:
                    ch_names.append(f"{fname}_{cn}")
            new_channels = ChannelIndex(ch_names)
        else:
            new_channels = None

        return X.with_data(stacked, channels=new_channels)


# ======================================================================
# FrameFeature implementations
# ======================================================================

class ZCR(FrameFeature):
    """Zero-crossing rate per frame."""

    names = ["zcr"]
    needs: set[str] = set()

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        # frames: (N, W, C)
        signs = np.sign(frames[:, 1:, :]) != np.sign(frames[:, :-1, :])
        zcr = np.mean(signs.astype(np.float64), axis=1)  # (N, C)
        if zcr.shape[-1] == 1:
            return zcr.squeeze(-1)  # (N,) for single channel
        return zcr.reshape(zcr.shape[0], -1)


class RMS(FrameFeature):
    """Root mean square energy per frame."""

    names = ["rms"]
    needs: set[str] = set()

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        rms = np.sqrt(np.mean(frames**2, axis=1))  # (N, C)
        if rms.shape[-1] == 1:
            return rms.squeeze(-1)
        return rms.reshape(rms.shape[0], -1)


class SpectralEntropy(FrameFeature):
    """Spectral entropy per frame."""

    names = ["spectral_entropy"]
    needs = {"power"}

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        power = cache["power"]  # (N, F, C)
        psd = power / (np.sum(power, axis=1, keepdims=True) + 1e-12)
        entropy = -np.sum(psd * np.log2(psd + 1e-12), axis=1)  # (N, C)
        if entropy.shape[-1] == 1:
            return entropy.squeeze(-1)
        return entropy.reshape(entropy.shape[0], -1)


class SpectralCentroid(FrameFeature):
    """Spectral centroid per frame."""

    names = ["spectral_centroid"]
    needs = {"magnitude"}

    def __init__(self, n_fft: int = 512, sr: float = 16000) -> None:
        self.n_fft = n_fft
        self.sr = sr
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        self._freqs = freqs

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        mag = cache["magnitude"]  # (N, F, C)
        freqs = self._freqs[:mag.shape[1]]
        centroid = (
            np.sum(mag * freqs[np.newaxis, :, np.newaxis], axis=1)
            / (np.sum(mag, axis=1) + 1e-12)
        )  # (N, C)
        if centroid.shape[-1] == 1:
            return centroid.squeeze(-1)
        return centroid.reshape(centroid.shape[0], -1)


class SpectralRolloff(FrameFeature):
    """Spectral rolloff frequency per frame."""

    names = ["spectral_rolloff"]
    needs = {"magnitude"}

    def __init__(self, n_fft: int = 512, sr: float = 16000, roll_percent: float = 0.85) -> None:
        self.n_fft = n_fft
        self.sr = sr
        self.roll_percent = roll_percent
        self._freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        mag = cache["magnitude"]  # (N, F, C)
        total = np.sum(mag, axis=1, keepdims=True)  # (N, 1, C)
        cumsum = np.cumsum(mag, axis=1)  # (N, F, C)
        threshold = self.roll_percent * total

        # Find first bin where cumsum exceeds threshold
        mask = cumsum >= threshold  # (N, F, C)
        F = mag.shape[1]
        freqs = self._freqs[:F]

        # argmax on bool gives first True
        rolloff_idx = np.argmax(mask, axis=1)  # (N, C)
        rolloff = freqs[rolloff_idx]  # (N, C)

        if rolloff.shape[-1] == 1:
            return rolloff.squeeze(-1)
        return rolloff.reshape(rolloff.shape[0], -1)


class SpectralFlatness(FrameFeature):
    """Spectral flatness (Wiener entropy) per frame."""

    names = ["spectral_flatness"]
    needs = {"magnitude"}

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        mag = cache["magnitude"] + 1e-12  # (N, F, C)
        geo_mean = np.exp(np.mean(np.log(mag), axis=1))  # (N, C)
        arith_mean = np.mean(mag, axis=1)  # (N, C)
        flatness = geo_mean / (arith_mean + 1e-12)  # (N, C)
        if flatness.shape[-1] == 1:
            return flatness.squeeze(-1)
        return flatness.reshape(flatness.shape[0], -1)


class MFCC(FrameFeature):
    """Mel-frequency cepstral coefficients per frame."""

    needs = {"power"}

    def __init__(self, n_mfcc: int = 13, n_mels: int = 64, sr: float = 16000, n_fft: int = 512) -> None:
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.sr = sr
        self.n_fft = n_fft
        self.names = [f"mfcc_{i}" for i in range(n_mfcc)]
        self._mel_basis = self._build_mel_filter()

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filter(self) -> np.ndarray:
        fmax = self.sr / 2.0
        F = self.n_fft // 2 + 1
        mel_min = self._hz_to_mel(0)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bins = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        filters = np.zeros((self.n_mels, F))
        for i in range(self.n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)
        return filters

    def extract(self, frames: np.ndarray, cache: dict) -> np.ndarray:
        power = cache["power"]  # (N, F, C)
        N, F_bins, C = power.shape

        results = []
        for c in range(C):
            mel_spec = self._mel_basis[:, :F_bins] @ power[:, :, c].T  # (n_mels, N)
            log_mel = np.log(mel_spec + 1e-12)  # (n_mels, N)

            # DCT-II (manual, avoids scipy dependency for this)
            n = np.arange(self.n_mfcc)[:, np.newaxis]
            k = np.arange(self.n_mels)[np.newaxis, :]
            dct_matrix = np.cos(np.pi * n * (2 * k + 1) / (2 * self.n_mels))
            mfccs = dct_matrix @ log_mel  # (n_mfcc, N)
            results.append(mfccs.T)  # (N, n_mfcc)

        if C == 1:
            return results[0]
        return np.concatenate(results, axis=-1)  # (N, n_mfcc * C)
