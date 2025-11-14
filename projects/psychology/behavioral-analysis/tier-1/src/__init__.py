"""
Multi-Modal Affect Recognition Toolkit for SageMaker Studio Lab

Utilities for processing EEG, facial, and physiological signals for emotion recognition.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .eeg_utils import (
    preprocess_eeg,
    extract_spectral_features,
    compute_eeg_connectivity,
)
from .facial_utils import (
    extract_facial_features,
    detect_emotions_from_face,
    compute_action_units,
)
from .physio_utils import (
    preprocess_physiological,
    compute_hrv,
    extract_scr_features,
    extract_respiratory_features,
)
from .fusion import (
    early_fusion,
    late_fusion,
    hybrid_fusion,
)
from .visualization import (
    plot_eeg_topography,
    plot_facial_landmarks,
    plot_multimodal_timeline,
    plot_fusion_performance,
)

__all__ = [
    "preprocess_eeg",
    "extract_spectral_features",
    "compute_eeg_connectivity",
    "extract_facial_features",
    "detect_emotions_from_face",
    "compute_action_units",
    "preprocess_physiological",
    "compute_hrv",
    "extract_scr_features",
    "extract_respiratory_features",
    "early_fusion",
    "late_fusion",
    "hybrid_fusion",
    "plot_eeg_topography",
    "plot_facial_landmarks",
    "plot_multimodal_timeline",
    "plot_fusion_performance",
]
