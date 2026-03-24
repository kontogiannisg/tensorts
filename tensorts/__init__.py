"""
tensorts — A channel-aware, shape-flexible tensor pipeline for multivariate time series.

Core:
    TimeTensor, Dim, DimSchema, ChannelIndex

Pipeline:
    Pipeline, Transform, ChannelTransform, ChannelSelect

Transforms:
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
    Interpolate, LabelEncoder,
    SlidingWindow, Pad, Truncate, Resample,
    PreEmphasis, ButterworthFilter, BandpassFilter, STFT, MelSpectrogram, LogTransform,
    FunctionTransform, StatefulFunctionTransform,
    FeatureExtractor, make_statistical_extractor,
    Flatten, Reshape, Transpose, PCA, Squeeze, Unsqueeze

Sklearn integration:
    SklearnWrapper — wraps a tensorts Pipeline + sklearn estimator for use with
    cross_val_score, GridSearchCV, cross_validate, etc.

Utilities:
    temporal_split, stratified_temporal_split, kfold_temporal,
    from_numpy, from_csv, from_dataframe, to_numpy, to_sklearn, to_dataframe
"""

__version__ = "0.1.0"

# Core
# Sklearn integration
from .compat import SklearnWrapper
from .core import ChannelIndex, Dim, DimSchema, TimeTensor

# IO
from .io import from_csv, from_dataframe, from_numpy, to_dataframe, to_numpy, to_sklearn

# Transforms (public API)
from .transforms import (
    PCA,
    STFT,
    BandpassFilter,
    ButterworthFilter,
    ChannelSelect,
    ChannelTransform,
    FeatureExtractor,
    Flatten,
    FunctionTransform,
    Interpolate,
    LabelEncoder,
    LogTransform,
    MaxAbsScaler,
    MelSpectrogram,
    MinMaxScaler,
    Normalizer,
    Pad,
    Pipeline,
    PowerTransformer,
    PreEmphasis,
    QuantileTransformer,
    Resample,
    Reshape,
    RobustScaler,
    Scope,
    SlidingWindow,
    Squeeze,
    StandardScaler,
    StatefulFunctionTransform,
    Transform,
    Transpose,
    Truncate,
    Unsqueeze,
    make_statistical_extractor,
)

# Utilities
from .utils import kfold_temporal, stratified_temporal_split, temporal_split
