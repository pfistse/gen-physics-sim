"""
Models package for generative physics simulation.

This package contains PyTorch Lightning modules for different generative models
used in physics simulation tasks.
"""

from .flow_matching import FlowMatchingModel
from .consistency_model import ConsistencyModel
from .diffusion_model import DiffusionModel

__all__ = ["FlowMatchingModel", "ConsistencyModel", "DiffusionModel"]
