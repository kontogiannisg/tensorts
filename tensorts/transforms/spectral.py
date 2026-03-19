"""
Spectral transforms: FFT, STFT, mel spectrograms, digital filters.

All operations are vectorized over batch and channel dimensions.
Filter transforms use scipy.signal internally.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.channels import ChannelIndex
from ..core.dims import Dim
from ..core.tensor import TimeTensor
from .base import Scope, Transform


# ======================================================================
# PreEmphasis
# ======================================================================

class PreEmphasis(Transform):
    """Pre-emphasis filter: ``y[t] = x[t] - coeff * x[t-1]``.

    Common first step in speech/audio pipelines to boost high frequencies.
    """

    scope = Scope.ELEMENTWISE
    supports_ragged = True

    def __init__(self, coeff: float = 0.97) -> None:
        super().__init__()
        self.coeff = coeff

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "PreEmphasis":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        if X.is_ragged:
            new_samples = []
            for s in X._samples:
                emphasized = np.empty_like(s)
                emphasized[0] = s[0]
                emphasized[1:] = s[1:] - self.coeff * s[:-1]
                new_samples.append(emphasized)
            return X.with_ragged(new_samples)

        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        data = X.data
        # Shift along time axis
        slices_curr = [slice(None)] * data.ndim
        slices_prev = [slice(None)] * data.ndim
        slices_curr[t_ax] = slice(1, None)
        slices_prev[t_ax] = slice(None, -1)

        result = data.copy()
        result[tuple(slices_curr)] = (
            data[tuple(slices_curr)] - self.coeff * data[tuple(slices_prev)]
        )
        return X.with_data(result)


# ======================================================================
# ButterworthFilter
# ======================================================================

class ButterworthFilter(Transform):
    """Butterworth IIR filter along the time axis.

    Parameters
    ----------
    cutoff : float or tuple of float
        Cutoff frequency (Hz).  For bandpass/bandstop, provide (low, high).
    order : int
        Filter order.
    btype : str
        ``'lowpass'``, ``'highpass'``, ``'bandpass'``, ``'bandstop'``.
    sampling_rate : float or None
        If None, reads from ``TimeTensor.metadata['sampling_rate']``.
    """

    scope = Scope.PER_CHANNEL
    supports_ragged = True

    def __init__(
        self,
        cutoff: float | tuple[float, float] = 20.0,
        order: int = 4,
        btype: str = "lowpass",
        sampling_rate: float | None = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.order = order
        self.btype = btype
        self.sampling_rate = sampling_rate
        self._sos: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "ButterworthFilter":
        from scipy.signal import butter

        sr = self.sampling_rate or X.metadata.get("sampling_rate")
        if sr is None:
            raise ValueError(
                "sampling_rate not provided and not found in metadata. "
                "Pass sampling_rate= or set metadata={'sampling_rate': ...}."
            )
        nyq = sr / 2.0
        if isinstance(self.cutoff, (list, tuple)):
            Wn = [c / nyq for c in self.cutoff]
        else:
            Wn = self.cutoff / nyq

        self._sos = butter(self.order, Wn, btype=self.btype, output="sos")
        self._is_fitted = True
        return self

    def _apply_filter(self, arr: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfiltfilt
        return sosfiltfilt(self._sos, arr, axis=0).astype(arr.dtype)

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        if X.is_ragged:
            new_samples = [self._apply_filter(s) for s in X._samples]
            return X.with_ragged(new_samples)

        t_ax = X.axis(Dim.TIME) if X.has_dim(Dim.TIME) else 1
        from scipy.signal import sosfiltfilt
        filtered = sosfiltfilt(self._sos, X.data, axis=t_ax).astype(X.dtype)
        return X.with_data(filtered)


# ======================================================================
# STFT
# ======================================================================

class STFT(Transform):
    """Short-Time Fourier Transform.

    Input:  (B, T, C)
    Output: (B, FREQUENCY, FRAME, C) — magnitude spectrogram.

    Parameters
    ----------
    n_fft : int
        FFT window size.
    hop_length : int or None
        Hop between frames.  Defaults to ``n_fft // 4``.
    window : str
        Window function name (``'hann'``, ``'hamming'``, etc.).
    power : float or None
        Exponent for magnitude.  1.0 = amplitude, 2.0 = power, None = complex.
    """

    scope = Scope.PER_CHANNEL

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int | None = None,
        window: str = "hann",
        power: float | None = 2.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.window = window
        self.power = power
        self._window_func: np.ndarray | None = None

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "STFT":
        from scipy.signal import get_window
        self._window_func = get_window(self.window, self.n_fft).astype(np.float64)
        self._is_fitted = True
        return self

    def _stft_2d(self, arr: np.ndarray) -> np.ndarray:
        """STFT on a (T, C) array → (F, N_frames, C)."""
        T, C = arr.shape
        n_frames = 1 + (T - self.n_fft) // self.hop_length
        F = self.n_fft // 2 + 1

        result = np.empty((F, n_frames, C), dtype=np.float64 if self.power else np.complex128)
        for c in range(C):
            for i in range(n_frames):
                start = i * self.hop_length
                frame = arr[start : start + self.n_fft, c] * self._window_func
                spectrum = np.fft.rfft(frame)
                if self.power is not None:
                    result[:, i, c] = np.abs(spectrum) ** self.power
                else:
                    result[:, i, c] = spectrum
        return result

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        data = X.data
        t_ax = X.axis(Dim.TIME)
        ch_dim = X._find_channel_dim(X.schema)
        ch_ax = X.axis(ch_dim) if ch_dim is not None else data.ndim - 1

        # Move T and C to last
        batch_axes = [i for i in range(data.ndim) if i not in (t_ax, ch_ax)]
        perm = batch_axes + [t_ax, ch_ax]
        transposed = np.transpose(data, perm)
        leading = transposed.shape[:-2]
        flat = transposed.reshape(-1, transposed.shape[-2], transposed.shape[-1])

        results = np.stack([self._stft_2d(flat[i]) for i in range(flat.shape[0])], axis=0)
        # results: (N, F, N_frames, C)
        results = results.reshape(*leading, results.shape[-3], results.shape[-2], results.shape[-1])

        # Output dims: replace TIME with FREQUENCY, TIME (frame-level)
        new_dims = []
        for d in X.dims:
            if d == Dim.TIME:
                new_dims.extend([Dim.FREQUENCY, Dim.TIME])
            else:
                new_dims.append(d)

        return X.with_data(results, dims=tuple(new_dims))


# ======================================================================
# MelSpectrogram
# ======================================================================

class MelSpectrogram(Transform):
    """Mel-scaled spectrogram.

    Input:  (B, T, C)
    Output: (B, MEL_BINS, FRAMES, C)

    Parameters
    ----------
    n_fft : int
    hop_length : int or None
    n_mels : int
        Number of Mel bands.
    sampling_rate : float or None
    fmin : float
    fmax : float or None
    power : float
    """

    scope = Scope.PER_CHANNEL

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int | None = None,
        n_mels: int = 64,
        sampling_rate: float | None = None,
        fmin: float = 0.0,
        fmax: float | None = None,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self._stft: STFT | None = None
        self._mel_basis: np.ndarray | None = None

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filter(self, sr: float) -> np.ndarray:
        """Build a (n_mels, n_fft//2+1) Mel filterbank matrix."""
        fmax = self.fmax or sr / 2.0
        F = self.n_fft // 2 + 1

        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((self.n_fft + 1) * hz_points / sr).astype(int)

        filters = np.zeros((self.n_mels, F))
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)
        return filters

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "MelSpectrogram":
        sr = self.sampling_rate or X.metadata.get("sampling_rate")
        if sr is None:
            raise ValueError("sampling_rate required.")

        self._stft = STFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=self.power,
        )
        self._stft.fit(X)
        self._mel_basis = self._build_mel_filter(sr)
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        self._check_fitted()
        # First compute STFT → (B, F, N_frames, C)
        stft_result = self._stft.transform(X)
        spec = stft_result.data  # (..., F, N_frames, C)

        # Apply mel filterbank: mel_basis is (n_mels, F)
        # spec has F on axis -3
        # mel = mel_basis @ spec[..., :, frame, ch]
        # Use tensordot / einsum for vectorized operation
        # spec shape: (..., F, N_frames, C)
        mel = np.tensordot(self._mel_basis, spec, axes=(1, -3))
        # Result: (n_mels, ..., N_frames, C) — need to move n_mels
        # Actually tensordot sums over F, giving (n_mels, leading..., N_frames, C)
        # We want (..., n_mels, N_frames, C)
        mel = np.moveaxis(mel, 0, -3)

        # Replace FREQUENCY dim (which was from STFT) with FREQUENCY (mel bins)
        new_dims = list(stft_result.dims)  # same structure

        return stft_result.with_data(mel, dims=tuple(new_dims))


# ======================================================================
# LogTransform
# ======================================================================

class LogTransform(Transform):
    """Element-wise log transform: ``log(x + eps)``."""

    scope = Scope.ELEMENTWISE
    supports_ragged = True

    def __init__(self, eps: float = 1e-10, base: str = "natural") -> None:
        super().__init__()
        self.eps = eps
        self.base = base

    def fit(self, X: TimeTensor, y: np.ndarray | None = None) -> "LogTransform":
        self._is_fitted = True
        return self

    def transform(self, X: TimeTensor) -> TimeTensor:
        log_fn = np.log if self.base == "natural" else np.log10
        if X.is_ragged:
            new_samples = [log_fn(s + self.eps) for s in X._samples]
            return X.with_ragged(new_samples)
        return X.with_data(log_fn(X.data + self.eps))


# ======================================================================
# BandpassFilter (convenience wrapper)
# ======================================================================

class BandpassFilter(ButterworthFilter):
    """Convenience bandpass filter."""

    def __init__(
        self,
        low: float,
        high: float,
        order: int = 4,
        sampling_rate: float | None = None,
    ) -> None:
        super().__init__(
            cutoff=(low, high),
            order=order,
            btype="bandpass",
            sampling_rate=sampling_rate,
        )
        self.low = low
        self.high = high
