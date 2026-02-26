"""
Phase 3: Training Techniques

This module provides implementations of training techniques for deep learning:
    - Normalization: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
    - Regularization: Dropout, L1/L2 regularization
    - Learning Rate Scheduling

Modules:
    normalization: 4 normalization techniques with forward/backward
"""

from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    InstanceNorm2d,
    GroupNorm,
    NORMALIZATION_FUNCTIONS,
    get_normalization,
)

__all__ = [
    # Normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "InstanceNorm2d",
    "GroupNorm",
    "NORMALIZATION_FUNCTIONS",
    "get_normalization",
]
