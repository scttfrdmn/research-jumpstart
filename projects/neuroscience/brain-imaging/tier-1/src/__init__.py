"""
Neuroscience Brain Imaging Analysis Suite
Tier 1: Multi-Subject Ensemble Connectivity Analysis

Modules:
- data_utils: Data loading and preprocessing
- connectivity: Functional connectivity analysis
- models: Deep learning architectures
- visualization: Brain and network plotting
"""

__version__ = "1.0.0"
__author__ = "Research Jumpstart"

from . import connectivity, data_utils, models, visualization

__all__ = ["connectivity", "data_utils", "models", "visualization"]
