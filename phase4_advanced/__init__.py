"""
Phase 4: Advanced Training Techniques

This module provides advanced training utilities:
    - Gradient stability and diagnostics
    - Mixed precision training
    - Training debugging and monitoring
    - Early stopping

Modules:
    gradient_stability: Gradient clipping, flow analysis, vanishing/explosion detection
"""

from .gradient_stability import (
    # Gradient clipping functions
    clip_grad_norm,
    clip_grad_value,
    # Diagnostics
    GradientFlowAnalyzer,
    GradientStats,
    detect_vanishing_gradient,
    detect_exploding_gradient,
    # Solutions
    apply_skip_connection,
    LayerScale,
)

__all__ = [
    # Clipping
    "clip_grad_norm",
    "clip_grad_value",
    # Diagnostics
    "GradientFlowAnalyzer",
    "GradientStats",
    "detect_vanishing_gradient",
    "detect_exploding_gradient",
    # Solutions
    "apply_skip_connection",
    "LayerScale",
]
