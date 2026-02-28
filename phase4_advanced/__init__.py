"""
Phase 4: Advanced Training Techniques

This module provides advanced training utilities:
    - Gradient stability and diagnostics
    - Mixed precision training (FP16/BF16/TF32)
    - Training debugging and monitoring
    - NaN loss debugging and recovery
    - Early stopping

Modules:
    gradient_stability: Gradient clipping, flow analysis, vanishing/explosion detection
    mixed_precision: AMP training, GradScaler, precision detection
    nan_debugger: NaN detection, training stability monitoring, auto recovery
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

from .mixed_precision import (
    # Precision detection
    is_fp16_supported,
    is_bf16_supported,
    is_tf32_supported,
    enable_tf32,
    get_recommended_precision,
    get_device_info,
    # Scaler
    GradScalerConfig,
    MixedPrecisionScaler,
    # Trainer
    MixedPrecisionTrainer,
    # Utilities
    compare_precision_modes,
    enable_optimizations_for_small_vram,
    get_precision_info,
    MIXED_PRECISION_MODES,
)

from .nan_debugger import (
    # Components
    NaNDebugger,
    TrainingStabilityMonitor,
    AutoRecoveryHandler,
    DataValidator,
    NumericalStabilityTester,
    # Enums
    StabilityStatus,
    RecoveryAction,
    # Data classes
    DiagnosticResult,
    StabilityReport,
    # Utilities
    safe_log,
    safe_exp,
    safe_divide,
    detect_anomaly,
    get_nan_debugger,
    NAN_DEBUGGER_COMPONENTS,
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
    # Mixed Precision
    "is_fp16_supported",
    "is_bf16_supported",
    "is_tf32_supported",
    "enable_tf32",
    "get_recommended_precision",
    "get_device_info",
    "GradScalerConfig",
    "MixedPrecisionScaler",
    "MixedPrecisionTrainer",
    "compare_precision_modes",
    "enable_optimizations_for_small_vram",
    "get_precision_info",
    "MIXED_PRECISION_MODES",
    # NaN Debugger
    "NaNDebugger",
    "TrainingStabilityMonitor",
    "AutoRecoveryHandler",
    "DataValidator",
    "NumericalStabilityTester",
    "StabilityStatus",
    "RecoveryAction",
    "DiagnosticResult",
    "StabilityReport",
    "safe_log",
    "safe_exp",
    "safe_divide",
    "detect_anomaly",
    "get_nan_debugger",
    "NAN_DEBUGGER_COMPONENTS",
]
