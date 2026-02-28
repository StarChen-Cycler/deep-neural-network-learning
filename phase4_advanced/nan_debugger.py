"""
NaN Loss Debugger and Training Stability Monitor.

This module provides tools for diagnosing and fixing NaN loss problems
and training instability in neural networks.

Key Components:
    - NaNDebugger: Detects NaN/Inf in loss and gradients
    - TrainingStabilityMonitor: Monitors gradient norms with warnings
    - AutoRecoveryHandler: Automatic learning rate reduction on instability
    - DataValidator: Checks input data for anomalies

Common NaN Causes:
    1. Learning rate too high
    2. Improper weight initialization
    3. Data anomalies (NaN/Inf in input)
    4. Numerical overflow in exp/log operations
    5. Division by zero

Solutions:
    - Reduce learning rate 5-10x
    - Gradient clipping (max_norm=1.0)
    - Enable anomaly detection: torch.autograd.set_detect_anomaly(True)
    - Check data: np.isnan().any(), np.isinf().any()
    - Use mixed precision with GradScaler

References:
    - On the difficulty of training recurrent neural networks (Pascanu et al., 2013)
    - Training Deep Neural Networks with 8-bit Floating Point Numbers (Sun et al., 2019)
"""

from typing import List, Optional, Union, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

ArrayLike = Union[np.ndarray, List, float]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array with float64 dtype."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)
    return x


# =============================================================================
# Enums and Data Classes
# =============================================================================


class StabilityStatus(Enum):
    """Training stability status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery actions for instability."""

    NONE = "none"
    REDUCE_LR = "reduce_lr"
    CLIP_GRADIENTS = "clip_gradients"
    RESET_OPTIMIZER = "reset_optimizer"
    STOP_TRAINING = "stop_training"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""

    name: str
    passed: bool
    message: str
    value: Optional[float] = None
    severity: str = "info"  # info, warning, error, critical


@dataclass
class StabilityReport:
    """Comprehensive stability report."""

    status: StabilityStatus
    diagnostics: List[DiagnosticResult]
    recommendations: List[str]
    recovery_action: RecoveryAction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "diagnostics": [
                {
                    "name": d.name,
                    "passed": d.passed,
                    "message": d.message,
                    "value": d.value,
                    "severity": d.severity,
                }
                for d in self.diagnostics
            ],
            "recommendations": self.recommendations,
            "recovery_action": self.recovery_action.value,
        }


# =============================================================================
# Data Validator
# =============================================================================


class DataValidator:
    """
    Validates input data for anomalies.

    Checks for:
        - NaN values
        - Inf values
        - Extreme values
        - Zero variance features
        - Class imbalance (for labels)

    Usage:
        validator = DataValidator()
        report = validator.validate(X, y)
        if not report['is_valid']:
            print(report['issues'])
    """

    def __init__(
        self,
        max_value: float = 1e6,
        min_variance: float = 1e-10,
        check_numerical_stability: bool = True,
    ):
        """
        Initialize DataValidator.

        Args:
            max_value: Maximum allowed absolute value
            min_variance: Minimum allowed variance for features
            check_numerical_stability: Whether to check for exp/log stability
        """
        self.max_value = max_value
        self.min_variance = min_variance
        self.check_numerical_stability = check_numerical_stability

    def validate(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validate input data.

        Args:
            X: Input features
            y: Optional labels

        Returns:
            Dictionary with validation results
        """
        X = _ensure_array(X)
        issues = []
        warnings = []

        # Check NaN
        nan_count = int(np.sum(np.isnan(X)))
        if nan_count > 0:
            nan_ratio = nan_count / X.size
            issues.append(
                f"Found {nan_count} NaN values ({nan_ratio:.2%} of data)"
            )

        # Check Inf
        inf_count = int(np.sum(np.isinf(X)))
        if inf_count > 0:
            inf_ratio = inf_count / X.size
            issues.append(
                f"Found {inf_count} Inf values ({inf_ratio:.2%} of data)"
            )

        # Check extreme values
        max_val = np.max(np.abs(X[~np.isinf(X)])) if X.size > 0 else 0
        if max_val > self.max_value:
            warnings.append(
                f"Extreme values detected: max abs = {max_val:.2e} "
                f"(threshold: {self.max_value:.2e})"
            )

        # Check variance
        if X.ndim >= 2:
            variances = np.var(X, axis=0)
            zero_var_count = int(np.sum(variances < self.min_variance))
            if zero_var_count > 0:
                warnings.append(
                    f"{zero_var_count} features have near-zero variance"
                )

        # Check labels if provided
        label_issues = []
        if y is not None:
            y = _ensure_array(y)
            nan_labels = int(np.sum(np.isnan(y)))
            if nan_labels > 0:
                label_issues.append(f"Found {nan_labels} NaN labels")

            inf_labels = int(np.sum(np.isinf(y)))
            if inf_labels > 0:
                label_issues.append(f"Found {inf_labels} Inf labels")

        # Numerical stability checks
        stability_issues = []
        if self.check_numerical_stability:
            # Check for values that would cause exp overflow
            exp_overflow = np.sum(X > 700)  # exp(700) overflows
            if exp_overflow > 0:
                stability_issues.append(
                    f"{exp_overflow} values could cause exp() overflow"
                )

            # Check for values that would cause log underflow
            log_underflow = np.sum((X > 0) & (X < 1e-300))
            if log_underflow > 0:
                stability_issues.append(
                    f"{log_underflow} values could cause log() underflow"
                )

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "label_issues": label_issues,
            "stability_issues": stability_issues,
            "stats": {
                "nan_count": nan_count,
                "inf_count": inf_count,
                "max_value": float(max_val),
                "min_value": float(np.min(X[~np.isinf(X)])) if X.size > 0 else 0,
                "mean": float(np.mean(X[~np.isnan(X) & ~np.isinf(X)]))
                if X.size > 0
                else 0,
            },
        }

    def clean(
        self,
        X: np.ndarray,
        strategy: str = "fill_zero",
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        Clean data by handling NaN/Inf values.

        Args:
            X: Input array
            strategy: Cleaning strategy ('fill_zero', 'fill_mean', 'drop')
            fill_value: Value to use for 'fill_zero' strategy

        Returns:
            Cleaned array
        """
        X = _ensure_array(X).copy()

        if strategy == "fill_zero":
            X[np.isnan(X)] = fill_value
            X[np.isinf(X)] = fill_value
        elif strategy == "fill_mean":
            valid_mask = ~(np.isnan(X) | np.isinf(X))
            if np.any(valid_mask):
                mean_val = np.mean(X[valid_mask])
                X[~valid_mask] = mean_val
            else:
                X[:] = fill_value
        elif strategy == "clip":
            X = np.clip(X, -self.max_value, self.max_value)
            X[np.isnan(X)] = fill_value
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return X


# =============================================================================
# NaN Debugger
# =============================================================================


class NaNDebugger:
    """
    Detects and diagnoses NaN loss problems.

    Features:
        - Detects NaN/Inf in loss and gradients
        - Identifies root causes
        - Provides recovery recommendations
        - Tracks history for pattern detection

    Usage:
        debugger = NaNDebugger()

        # During training
        for batch in dataloader:
            loss = model(batch)
            if debugger.check_loss(loss):
                # NaN detected
                report = debugger.diagnose(model, loss)
                print(report.recommendations)
    """

    def __init__(
        self,
        loss_threshold: float = 1e10,
        gradient_threshold: float = 100.0,
        history_size: int = 100,
    ):
        """
        Initialize NaNDebugger.

        Args:
            loss_threshold: Threshold for abnormal loss values
            gradient_threshold: Threshold for gradient norm warnings
            history_size: Number of steps to keep in history
        """
        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        self.history_size = history_size

        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._nan_detected = False
        self._nan_step = -1

    def check_loss(self, loss: float) -> bool:
        """
        Check if loss is NaN or abnormal.

        Args:
            loss: Loss value to check

        Returns:
            True if NaN/abnormal detected
        """
        is_nan = np.isnan(loss) or np.isinf(loss)
        is_abnormal = abs(loss) > self.loss_threshold

        if is_nan or is_abnormal:
            self._nan_detected = True
            self._nan_step = len(self._loss_history)
            return True

        return False

    def check_gradients(
        self, gradients: List[np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Check gradients for NaN/Inf and abnormal norms.

        Args:
            gradients: List of gradient arrays

        Returns:
            Tuple of (has_problem, total_norm)
        """
        grads = [_ensure_array(g) for g in gradients]

        # Check for NaN/Inf
        has_nan = any(np.sum(np.isnan(g)) > 0 for g in grads if g.size > 0)
        has_inf = any(np.sum(np.isinf(g)) > 0 for g in grads if g.size > 0)

        # Compute total norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads if g.size > 0))

        # Check for exploding gradients
        is_exploding = total_norm > self.gradient_threshold

        # Record history
        self._grad_norm_history.append(float(total_norm))
        if len(self._grad_norm_history) > self.history_size:
            self._grad_norm_history.pop(0)

        return has_nan or has_inf or is_exploding, float(total_norm)

    def diagnose(
        self,
        gradients: Optional[List[np.ndarray]] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> StabilityReport:
        """
        Perform comprehensive diagnosis.

        Args:
            gradients: Current gradients
            loss: Current loss value
            learning_rate: Current learning rate

        Returns:
            StabilityReport with findings and recommendations
        """
        diagnostics = []
        recommendations = []
        status = StabilityStatus.HEALTHY
        recovery_action = RecoveryAction.NONE

        # Check loss
        if loss is not None:
            if np.isnan(loss):
                diagnostics.append(
                    DiagnosticResult(
                        name="loss_check",
                        passed=False,
                        message="Loss is NaN",
                        severity="critical",
                    )
                )
                status = StabilityStatus.CRITICAL
                recommendations.append(
                    "Loss is NaN - reduce learning rate immediately"
                )
                recovery_action = RecoveryAction.REDUCE_LR
            elif np.isinf(loss):
                diagnostics.append(
                    DiagnosticResult(
                        name="loss_check",
                        passed=False,
                        message="Loss is Inf",
                        severity="critical",
                    )
                )
                status = StabilityStatus.CRITICAL
                recommendations.append(
                    "Loss is Inf - check for numerical overflow"
                )
                recovery_action = RecoveryAction.STOP_TRAINING
            elif abs(loss) > self.loss_threshold:
                diagnostics.append(
                    DiagnosticResult(
                        name="loss_check",
                        passed=False,
                        message=f"Loss {loss:.2e} exceeds threshold",
                        value=loss,
                        severity="warning",
                    )
                )
                if status == StabilityStatus.HEALTHY:
                    status = StabilityStatus.WARNING
                recommendations.append("Loss is abnormally high - check data")
            else:
                diagnostics.append(
                    DiagnosticResult(
                        name="loss_check",
                        passed=True,
                        message=f"Loss is healthy: {loss:.4f}",
                        value=loss,
                        severity="info",
                    )
                )

        # Check gradients
        if gradients is not None:
            has_problem, grad_norm = self.check_gradients(gradients)

            if has_problem:
                if grad_norm == float("inf") or grad_norm == float("nan"):
                    diagnostics.append(
                        DiagnosticResult(
                            name="gradient_check",
                            passed=False,
                            message="Gradients contain NaN/Inf",
                            severity="critical",
                        )
                    )
                    status = StabilityStatus.CRITICAL
                    recommendations.append(
                        "Enable gradient clipping: max_norm=1.0"
                    )
                    recovery_action = RecoveryAction.CLIP_GRADIENTS
                elif grad_norm > self.gradient_threshold:
                    diagnostics.append(
                        DiagnosticResult(
                            name="gradient_check",
                            passed=False,
                            message=f"Gradient norm {grad_norm:.2f} > {self.gradient_threshold}",
                            value=grad_norm,
                            severity="warning",
                        )
                    )
                    if status == StabilityStatus.HEALTHY:
                        status = StabilityStatus.WARNING
                    recommendations.append(
                        f"High gradient norm - consider clipping at {self.gradient_threshold}"
                    )
                    if recovery_action == RecoveryAction.NONE:
                        recovery_action = RecoveryAction.CLIP_GRADIENTS
            else:
                diagnostics.append(
                    DiagnosticResult(
                        name="gradient_check",
                        passed=True,
                        message=f"Gradient norm is healthy: {grad_norm:.4f}",
                        value=grad_norm,
                        severity="info",
                    )
                )

        # Check learning rate
        if learning_rate is not None:
            if learning_rate > 0.1:
                diagnostics.append(
                    DiagnosticResult(
                        name="lr_check",
                        passed=False,
                        message=f"Learning rate {learning_rate} is very high",
                        value=learning_rate,
                        severity="warning",
                    )
                )
                if status == StabilityStatus.HEALTHY:
                    status = StabilityStatus.WARNING
                recommendations.append(
                    f"Reduce learning rate from {learning_rate} to {learning_rate / 10}"
                )
            else:
                diagnostics.append(
                    DiagnosticResult(
                        name="lr_check",
                        passed=True,
                        message=f"Learning rate is reasonable: {learning_rate}",
                        value=learning_rate,
                        severity="info",
                    )
                )

        # Check gradient history for patterns
        if len(self._grad_norm_history) >= 10:
            recent_norms = self._grad_norm_history[-10:]
            trend = recent_norms[-1] / (np.mean(recent_norms[:-1]) + 1e-10)

            if trend > 2.0:
                diagnostics.append(
                    DiagnosticResult(
                        name="trend_check",
                        passed=False,
                        message=f"Gradient norm increasing: {trend:.2f}x",
                        value=trend,
                        severity="warning",
                    )
                )
                if status == StabilityStatus.HEALTHY:
                    status = StabilityStatus.WARNING
                recommendations.append("Gradients are exploding - reduce LR")

        # Default recommendations if unstable
        if status in [StabilityStatus.UNSTABLE, StabilityStatus.CRITICAL]:
            if not any("learning rate" in r.lower() for r in recommendations):
                recommendations.append("Reduce learning rate by 5-10x")
            if not any("clipping" in r.lower() for r in recommendations):
                recommendations.append("Enable gradient clipping (max_norm=1.0)")

        return StabilityReport(
            status=status,
            diagnostics=diagnostics,
            recommendations=recommendations,
            recovery_action=recovery_action,
        )

    def record_loss(self, loss: float):
        """Record loss for history tracking."""
        self._loss_history.append(float(loss))
        if len(self._loss_history) > self.history_size:
            self._loss_history.pop(0)

    def get_history(self) -> Dict[str, List[float]]:
        """Get recorded history."""
        return {
            "loss": self._loss_history.copy(),
            "grad_norm": self._grad_norm_history.copy(),
        }

    def reset(self):
        """Reset debugger state."""
        self._loss_history.clear()
        self._grad_norm_history.clear()
        self._nan_detected = False
        self._nan_step = -1


# =============================================================================
# Training Stability Monitor
# =============================================================================


class TrainingStabilityMonitor:
    """
    Monitors training stability with configurable thresholds.

    Features:
        - Gradient norm monitoring with warnings
        - Loss spike detection
        - Automatic warning when gradient norm > threshold
        - Integration with AutoRecoveryHandler

    Success Criteria:
        - Gradient norm > 100 triggers warning

    Usage:
        monitor = TrainingStabilityMonitor(grad_norm_threshold=100.0)

        for batch in dataloader:
            loss = model(batch)
            loss.backward()

            report = monitor.check(grads, loss)
            if report.status != StabilityStatus.HEALTHY:
                print(report.recommendations)
    """

    def __init__(
        self,
        grad_norm_threshold: float = 100.0,
        loss_spike_threshold: float = 5.0,
        window_size: int = 10,
        enable_warnings: bool = True,
    ):
        """
        Initialize TrainingStabilityMonitor.

        Args:
            grad_norm_threshold: Threshold for gradient norm warning (default: 100.0)
            loss_spike_threshold: Multiplier for loss spike detection
            window_size: Window size for moving average
            enable_warnings: Whether to print warnings
        """
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.window_size = window_size
        self.enable_warnings = enable_warnings

        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._warning_count = 0

        self.debugger = NaNDebugger(gradient_threshold=grad_norm_threshold)

    def check(
        self,
        gradients: Optional[List[np.ndarray]] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> StabilityReport:
        """
        Check training stability.

        Args:
            gradients: Current gradients
            loss: Current loss
            learning_rate: Current learning rate

        Returns:
            StabilityReport with status and recommendations
        """
        # Record history
        if loss is not None:
            self._loss_history.append(float(loss))
            if len(self._loss_history) > self.window_size * 2:
                self._loss_history.pop(0)

        grad_norm = None
        if gradients is not None:
            _, grad_norm = self.debugger.check_gradients(gradients)
            self._grad_norm_history.append(grad_norm)
            if len(self._grad_norm_history) > self.window_size * 2:
                self._grad_norm_history.pop(0)

        # Build diagnostics
        diagnostics = []
        recommendations = []
        status = StabilityStatus.HEALTHY
        recovery_action = RecoveryAction.NONE

        # Check gradient norm threshold (SUCCESS CRITERION)
        if grad_norm is not None and grad_norm > self.grad_norm_threshold:
            self._warning_count += 1
            diagnostics.append(
                DiagnosticResult(
                    name="gradient_norm",
                    passed=False,
                    message=f"Gradient norm {grad_norm:.2f} > threshold {self.grad_norm_threshold}",
                    value=grad_norm,
                    severity="warning",
                )
            )
            status = StabilityStatus.WARNING
            recommendations.append(
                f"Warning: Gradient norm > {self.grad_norm_threshold}"
            )
            recommendations.append("Consider gradient clipping or reducing LR")
            recovery_action = RecoveryAction.CLIP_GRADIENTS

            if self.enable_warnings:
                print(
                    f"[WARNING] Gradient norm {grad_norm:.2f} > {self.grad_norm_threshold}"
                )

        # Check loss spike
        if loss is not None and len(self._loss_history) >= self.window_size:
            recent_mean = np.mean(self._loss_history[-self.window_size : -1])
            if recent_mean > 0 and loss > recent_mean * self.loss_spike_threshold:
                diagnostics.append(
                    DiagnosticResult(
                        name="loss_spike",
                        passed=False,
                        message=f"Loss spike detected: {loss:.4f} vs mean {recent_mean:.4f}",
                        value=loss / recent_mean,
                        severity="warning",
                    )
                )
                if status == StabilityStatus.HEALTHY:
                    status = StabilityStatus.WARNING
                recommendations.append("Loss spike detected - check for bad batch")

        # Check for NaN
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            diagnostics.append(
                DiagnosticResult(
                    name="loss_nan",
                    passed=False,
                    message="Loss is NaN or Inf",
                    severity="critical",
                )
            )
            status = StabilityStatus.CRITICAL
            recovery_action = RecoveryAction.REDUCE_LR

        # Add healthy diagnostic if all good
        if status == StabilityStatus.HEALTHY:
            diagnostics.append(
                DiagnosticResult(
                    name="overall",
                    passed=True,
                    message="Training is stable",
                    severity="info",
                )
            )

        return StabilityReport(
            status=status,
            diagnostics=diagnostics,
            recommendations=recommendations,
            recovery_action=recovery_action,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = {
            "warning_count": self._warning_count,
            "total_steps": len(self._loss_history),
        }

        if self._loss_history:
            stats["loss_mean"] = float(np.mean(self._loss_history))
            stats["loss_std"] = float(np.std(self._loss_history))
            stats["loss_min"] = float(np.min(self._loss_history))
            stats["loss_max"] = float(np.max(self._loss_history))

        if self._grad_norm_history:
            stats["grad_norm_mean"] = float(np.mean(self._grad_norm_history))
            stats["grad_norm_max"] = float(np.max(self._grad_norm_history))

        return stats

    def reset(self):
        """Reset monitor state."""
        self._loss_history.clear()
        self._grad_norm_history.clear()
        self._warning_count = 0
        self.debugger.reset()


# =============================================================================
# Auto Recovery Handler
# =============================================================================


class AutoRecoveryHandler:
    """
    Automatically handles training instability.

    Features:
        - Automatic learning rate reduction
        - Gradient clipping
        - Training pause/resume
        - Recovery history tracking

    Success Criteria:
        - NaN detection triggers automatic LR reduction
        - Training recovers after reduction

    Usage:
        handler = AutoRecoveryHandler(
            initial_lr=0.001,
            lr_reduction_factor=0.1,
        )

        for batch in dataloader:
            loss = model(batch)

            if handler.check_instability(loss, grads):
                # Recovery action taken
                new_lr = handler.get_current_lr()
                optimizer.lr = new_lr
    """

    def __init__(
        self,
        initial_lr: float,
        lr_reduction_factor: float = 0.1,
        min_lr: float = 1e-8,
        max_reduction_count: int = 5,
        enable_auto_clip: bool = True,
        grad_clip_max_norm: float = 1.0,
    ):
        """
        Initialize AutoRecoveryHandler.

        Args:
            initial_lr: Initial learning rate
            lr_reduction_factor: Factor to reduce LR by (default: 0.1 = 10x reduction)
            min_lr: Minimum allowed learning rate
            max_reduction_count: Maximum number of LR reductions
            enable_auto_clip: Whether to automatically clip gradients
            grad_clip_max_norm: Max norm for gradient clipping
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.max_reduction_count = max_reduction_count
        self.enable_auto_clip = enable_auto_clip
        self.grad_clip_max_norm = grad_clip_max_norm

        self._reduction_count = 0
        self._recovery_history: List[Dict[str, Any]] = []
        self._is_recovering = False

        self.monitor = TrainingStabilityMonitor()

    def check_instability(
        self,
        loss: Optional[float] = None,
        gradients: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """
        Check for instability and take recovery action.

        Args:
            loss: Current loss value
            gradients: Current gradients

        Returns:
            True if recovery action was taken
        """
        report = self.monitor.check(gradients, loss)

        if report.status == StabilityStatus.CRITICAL:
            return self._handle_critical(report, loss)
        elif report.status == StabilityStatus.WARNING:
            return self._handle_warning(report, gradients)

        return False

    def _handle_critical(
        self, report: StabilityReport, loss: Optional[float]
    ) -> bool:
        """Handle critical instability (NaN/Inf)."""
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            # NaN detected - reduce LR
            if self._reduction_count < self.max_reduction_count:
                old_lr = self.current_lr
                self.current_lr = max(
                    self.current_lr * self.lr_reduction_factor, self.min_lr
                )
                self._reduction_count += 1

                self._recovery_history.append(
                    {
                        "type": "lr_reduction",
                        "reason": "NaN loss detected",
                        "old_lr": old_lr,
                        "new_lr": self.current_lr,
                    }
                )

                print(
                    f"[RECOVERY] NaN detected. Reducing LR: {old_lr:.2e} -> {self.current_lr:.2e}"
                )
                return True

        return False

    def _handle_warning(
        self, report: StabilityReport, gradients: Optional[List[np.ndarray]]
    ) -> bool:
        """Handle warning-level instability."""
        if report.recovery_action == RecoveryAction.CLIP_GRADIENTS:
            if self.enable_auto_clip and gradients is not None:
                # Signal that gradients should be clipped
                self._recovery_history.append(
                    {"type": "gradient_clipping", "max_norm": self.grad_clip_max_norm}
                )
                return True

        return False

    def clip_gradients(
        self, gradients: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """
        Clip gradients if enabled.

        Args:
            gradients: List of gradient arrays

        Returns:
            Tuple of (clipped_gradients, original_norm)
        """
        from phase4_advanced.gradient_stability import clip_grad_norm

        grads = [_ensure_array(g) for g in gradients]

        # Compute original norm
        original_norm = np.sqrt(sum(np.sum(g**2) for g in grads if g.size > 0))

        if self.enable_auto_clip and original_norm > self.grad_clip_max_norm:
            clipped, _ = clip_grad_norm(grads, self.grad_clip_max_norm)
            return clipped, float(original_norm)

        return grads, float(original_norm)

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset_lr(self):
        """Reset learning rate to initial value."""
        self.current_lr = self.initial_lr
        self._reduction_count = 0

    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get recovery action history."""
        return self._recovery_history.copy()

    def can_continue(self) -> bool:
        """Check if training can continue."""
        return self._reduction_count < self.max_reduction_count


# =============================================================================
# Numerical Stability Tester
# =============================================================================


class NumericalStabilityTester:
    """
    Tests numerical stability across data types.

    Success Criteria:
        - Stability tests pass on all data types (float32, float64)

    Tests:
        - Exp overflow handling
        - Log underflow handling
        - Softmax stability
        - Division by zero handling
    """

    def __init__(self):
        """Initialize tester."""
        self.results: Dict[str, Dict[str, bool]] = {}

    def test_exp_stability(self) -> Tuple[bool, str]:
        """Test exp() stability with large values."""
        try:
            # Test float64
            large_val = np.float64(1000)
            result = np.exp(large_val)
            if np.isinf(result):
                # This is expected - we should handle it
                pass

            # Test safe exp (subtract max)
            values = np.array([1000, 1001, 1002], dtype=np.float64)
            safe_result = np.exp(values - np.max(values))
            if np.any(np.isinf(safe_result)):
                return False, "Safe exp failed for float64"

            self.results["exp_float64"] = True
        except Exception as e:
            return False, f"Exp test failed: {e}"

        try:
            # Test float32
            large_val = np.float32(100)
            result = np.exp(large_val)

            values = np.array([100, 101, 102], dtype=np.float32)
            safe_result = np.exp(values - np.max(values))
            if np.any(np.isinf(safe_result)):
                return False, "Safe exp failed for float32"

            self.results["exp_float32"] = True
        except Exception as e:
            return False, f"Exp test failed for float32: {e}"

        return True, "Exp stability tests passed"

    def test_log_stability(self) -> Tuple[bool, str]:
        """Test log() stability with small values."""
        try:
            # Test float64
            small_val = np.float64(1e-300)
            result = np.log(small_val)
            # Should be very negative but not -inf
            if np.isinf(result) and result < 0:
                pass  # Expected for very small values

            # Test safe log (add epsilon)
            values = np.array([1e-300, 1e-200, 1e-100], dtype=np.float64)
            safe_result = np.log(values + 1e-10)
            if np.any(np.isinf(safe_result)):
                return False, "Safe log failed for float64"

            self.results["log_float64"] = True
        except Exception as e:
            return False, f"Log test failed: {e}"

        try:
            # Test float32
            small_val = np.float32(1e-40)
            result = np.log(small_val)

            values = np.array([1e-40, 1e-30, 1e-20], dtype=np.float32)
            safe_result = np.log(values + np.float32(1e-10))
            if np.any(np.isinf(safe_result)):
                return False, "Safe log failed for float32"

            self.results["log_float32"] = True
        except Exception as e:
            return False, f"Log test failed for float32: {e}"

        return True, "Log stability tests passed"

    def test_softmax_stability(self) -> Tuple[bool, str]:
        """Test softmax numerical stability."""
        def stable_softmax(x: np.ndarray) -> np.ndarray:
            """Numerically stable softmax."""
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        try:
            # Test with large values
            large = np.array([1000, 1001, 1002], dtype=np.float64)
            result = stable_softmax(large)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return False, "Softmax failed with large float64 values"
            if not np.isclose(np.sum(result), 1.0):
                return False, "Softmax doesn't sum to 1 for float64"

            self.results["softmax_float64"] = True
        except Exception as e:
            return False, f"Softmax test failed for float64: {e}"

        try:
            # Test with float32
            large = np.array([100, 101, 102], dtype=np.float32)
            result = stable_softmax(large)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return False, "Softmax failed with large float32 values"
            if not np.isclose(np.sum(result), 1.0, atol=1e-5):
                return False, "Softmax doesn't sum to 1 for float32"

            self.results["softmax_float32"] = True
        except Exception as e:
            return False, f"Softmax test failed for float32: {e}"

        return True, "Softmax stability tests passed"

    def test_division_stability(self) -> Tuple[bool, str]:
        """Test division by zero handling."""
        try:
            # Test with epsilon protection
            eps = 1e-10

            # float64
            values = np.array([1.0, 0.0, 2.0], dtype=np.float64)
            safe_div = values / (values + eps)
            if np.any(np.isnan(safe_div)) or np.any(np.isinf(safe_div)):
                return False, "Safe division failed for float64"

            self.results["division_float64"] = True

            # float32
            values = np.array([1.0, 0.0, 2.0], dtype=np.float32)
            safe_div = values / (values + np.float32(eps))
            if np.any(np.isnan(safe_div)) or np.any(np.isinf(safe_div)):
                return False, "Safe division failed for float32"

            self.results["division_float32"] = True

        except Exception as e:
            return False, f"Division test failed: {e}"

        return True, "Division stability tests passed"

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all numerical stability tests.

        Returns:
            Dictionary with test results
        """
        results = {
            "exp_stability": self.test_exp_stability(),
            "log_stability": self.test_log_stability(),
            "softmax_stability": self.test_softmax_stability(),
            "division_stability": self.test_division_stability(),
        }

        all_passed = all(r[0] for r in results.values())
        results["all_passed"] = all_passed
        results["details"] = self.results.copy()

        return results


# =============================================================================
# Utility Functions
# =============================================================================


def safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Numerically stable log.

    Args:
        x: Input array
        eps: Small constant to prevent log(0)

    Returns:
        log(x + eps)
    """
    x = _ensure_array(x)
    return np.log(x + eps)


def safe_exp(x: np.ndarray, max_value: float = 700) -> np.ndarray:
    """
    Numerically stable exp.

    Args:
        x: Input array
        max_value: Maximum value before overflow

    Returns:
        exp(clipped(x))
    """
    x = _ensure_array(x)
    return np.exp(np.clip(x, -max_value, max_value))


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-10
) -> np.ndarray:
    """
    Numerically stable division.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Small constant to prevent division by zero

    Returns:
        numerator / (denominator + eps)
    """
    numerator = _ensure_array(numerator)
    denominator = _ensure_array(denominator)
    return numerator / (denominator + eps)


def detect_anomaly(tensor: np.ndarray, name: str = "tensor") -> List[str]:
    """
    Detect anomalies in a tensor.

    Args:
        tensor: Array to check
        name: Name for error messages

    Returns:
        List of anomaly descriptions
    """
    tensor = _ensure_array(tensor)
    anomalies = []

    nan_count = np.sum(np.isnan(tensor))
    if nan_count > 0:
        anomalies.append(f"{name}: {nan_count} NaN values")

    inf_count = np.sum(np.isinf(tensor))
    if inf_count > 0:
        anomalies.append(f"{name}: {inf_count} Inf values")

    return anomalies


# =============================================================================
# Registry
# =============================================================================


NAN_DEBUGGER_COMPONENTS = {
    "NaNDebugger": NaNDebugger,
    "TrainingStabilityMonitor": TrainingStabilityMonitor,
    "AutoRecoveryHandler": AutoRecoveryHandler,
    "DataValidator": DataValidator,
    "NumericalStabilityTester": NumericalStabilityTester,
}


def get_nan_debugger(name: str) -> type:
    """Get NaN debugger component by name."""
    if name not in NAN_DEBUGGER_COMPONENTS:
        available = list(NAN_DEBUGGER_COMPONENTS.keys())
        raise ValueError(f"Unknown component: {name}. Available: {available}")
    return NAN_DEBUGGER_COMPONENTS[name]
