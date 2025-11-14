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

from . import data_utils
from . import connectivity
from . import models
from . import visualization

__all__ = ['data_utils', 'connectivity', 'models', 'visualization']
