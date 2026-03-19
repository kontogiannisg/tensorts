"""
tensorts — A channel-aware, shape-flexible tensor pipeline for multivariate time series.

Core:
    TimeTensor, Dim, DimSchema, ChannelIndex

Pipeline:
    Pipeline, Transform, ChannelTransform, ChannelSelect

Transforms:
    StandardScaler, MinMaxScaler, RobustScaler, Interpolate, LabelEncoder,
    SlidingWindow, Pad, Truncate, Resample,
    PreEmphasis, ButterworthFilter, BandpassFilter, STFT, MelSpectrogram, LogTransform,
    StatisticalFeatures, RollingFeatures, FrameFeatureBank,
    Flatten, Reshape, Transpose, PCA, Squeeze, Unsqueeze

Utilities:
    temporal_split, stratified_temporal_split, kfold_temporal,
    from_numpy, from_csv, from_dataframe, to_numpy, to_sklearn, to_dataframe
"""

__version__ = "0.1.0"

# Core
from .core import ChannelIndex, Dim, DimSchema, TimeTensor

# Transforms (public API)
from .transforms import (
    MFCC,
    RMS,
    STFT,
    ZCR,
    BandpassFilter,
    ButterworthFilter,
    ChannelSelect,
    ChannelTransform,
    Flatten,
    FrameFeature,
    FrameFeatureBank,
    FunctionTransform,
    Interpolate,
    LabelEncoder,
    LogTransform,
    MelSpectrogram,
    MinMaxScaler,
    Pad,
    PCA,
    Pipeline,
    PreEmphasis,
    Resample,
    Reshape,
    RobustScaler,
    RollingFeatures,
    Scope,
    SlidingWindow,
    SpectralCentroid,
    SpectralEntropy,
    SpectralFlatness,
    SpectralRolloff,
    Squeeze,
    StandardScaler,
    StatefulFunctionTransform,
    StatisticalFeatures,
    Transform,
    Transpose,
    Truncate,
    Unsqueeze,
)

# IO
from .io import from_csv, from_dataframe, from_numpy, to_dataframe, to_numpy, to_sklearn

# Utilities
from .utils import kfold_temporal, stratified_temporal_split, temporal_split
