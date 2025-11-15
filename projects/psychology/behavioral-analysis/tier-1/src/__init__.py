"""
Multi-Modal Affect Recognition Toolkit for SageMaker Studio Lab

Utilities for processing EEG, facial, and physiological signals for emotion recognition.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .eeg_utils import (
    compute_eeg_connectivity,
    extract_spectral_features,
    preprocess_eeg,
)
from .facial_utils import (
    compute_action_units,
    detect_emotions_from_face,
    extract_facial_features,
)
from .fusion import (
    early_fusion,
    hybrid_fusion,
    late_fusion,
)
from .physio_utils import (
    compute_hrv,
    extract_respiratory_features,
    extract_scr_features,
    preprocess_physiological,
)
from .visualization import (
    plot_eeg_topography,
    plot_facial_landmarks,
    plot_fusion_performance,
    plot_multimodal_timeline,
)

__all__ = [
    "compute_action_units",
    "compute_eeg_connectivity",
    "compute_hrv",
    "detect_emotions_from_face",
    "early_fusion",
    "extract_facial_features",
    "extract_respiratory_features",
    "extract_scr_features",
    "extract_spectral_features",
    "hybrid_fusion",
    "late_fusion",
    "plot_eeg_topography",
    "plot_facial_landmarks",
    "plot_fusion_performance",
    "plot_multimodal_timeline",
    "preprocess_eeg",
    "preprocess_physiological",
]
