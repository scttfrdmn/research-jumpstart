"""
Multi-Modal Medical Imaging Ensemble
=====================================

Source code modules for medical image analysis across X-ray, CT, and MRI modalities.

Modules:
    data_utils: Data loading and preprocessing for all modalities
    preprocessing: Image preprocessing pipelines
    models: Deep learning model architectures
    ensemble: Ensemble learning methods
    evaluation: Clinical evaluation metrics
    visualization: Plotting and GradCAM interpretability
"""

__version__ = "1.0.0"
__author__ = "Research Jumpstart"

from . import data_utils, ensemble, evaluation, models, preprocessing, visualization

__all__ = ["data_utils", "ensemble", "evaluation", "models", "preprocessing", "visualization"]
