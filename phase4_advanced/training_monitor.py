"""
Training Monitor: Comprehensive training visualization and debugging.

This module provides:
    - TrainingMonitor: Unified monitoring for gradients, activations, weights
    - TensorBoard integration for gradient/weight histograms
    - Weight update rate tracking
    - Activation distribution monitoring

Theory:
    Gradient Monitoring:
        - Track gradient norm per layer to detect vanishing/exploding
        - Histogram analysis reveals dead neurons (spike at 0)
        - Update ratio = ||weight_update|| / ||weight|| should be ~0.001-0.01

    Activation Monitoring:
        - Dead neurons: high % of zeros in ReLU output
        - Saturation: tanh/sigmoid outputs clustered at extremes
        - Internal covariate shift: distribution changes over training

    Weight Update Monitoring:
        - Healthy ratio: 0.001 to 0.01 for SGD, up to 0.1 for Adam
        - Too low: learning rate too small or gradients vanishing
        - Too high: learning rate too large, risk of divergence

References:
    - Training with Active Focus on Dead Neurons (Lu et al., 2020)
    - Batch Normalization: Accelerating Deep Network Training (Ioffe & Szegedy, 2015)
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


class MonitorStatus(Enum):
    """Training monitor status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GradientReport:
    """Report on gradient statistics for a layer."""

    layer_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    norm: float
    zero_ratio: float
    nan_count: int
    inf_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_name": self.layer_name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "norm": self.norm,
            "zero_ratio": self.zero_ratio,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass
class ActivationReport:
    """Report on activation statistics for a layer."""

    layer_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    zero_ratio: float
    saturation_ratio: float  # For sigmoid/tanh: near 0 or 1 / -1 or 1
    nan_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_name": self.layer_name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "zero_ratio": self.zero_ratio,
            "saturation_ratio": self.saturation_ratio,
            "nan_count": self.nan_count,
        }


@dataclass
class WeightUpdateReport:
    """Report on weight update statistics."""

    layer_name: str
    weight_norm: float
    update_norm: float
    update_ratio: float
    gradient_norm: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_name": self.layer_name,
            "weight_norm": self.weight_norm,
            "update_norm": self.update_norm,
            "update_ratio": self.update_ratio,
            "gradient_norm": self.gradient_norm,
        }


@dataclass
class TrainingSnapshot:
    """Snapshot of training state at a step."""

    step: int
    loss: float
    learning_rate: float
    gradient_reports: List[GradientReport]
    activation_reports: List[ActivationReport]
    weight_update_reports: List[WeightUpdateReport]
    status: MonitorStatus
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gradient_reports": [r.to_dict() for r in self.gradient_reports],
            "activation_reports": [r.to_dict() for r in self.activation_reports],
            "weight_update_reports": [r.to_dict() for r in self.weight_update_reports],
            "status": self.status.value,
            "warnings": self.warnings,
        }


# =============================================================================
# Core Monitor Class
# =============================================================================


class TrainingMonitor:
    """
    Unified training monitor for gradients, activations, and weights.

    This class provides comprehensive monitoring of training progress
    with automatic detection of common issues like vanishing gradients,
    dead neurons, and inappropriate learning rates.

    Usage:
        monitor = TrainingMonitor()

        # During training loop
        for batch in dataloader:
            loss = model(batch)
            loss.backward()

            # Record gradients
            monitor.record_gradients(model)

            # Record activations (forward hook)
            monitor.record_activations(activations)

            # Check weight update ratio
            monitor.check_weight_updates(model, old_params, optimizer)

            # Get snapshot for logging
            snapshot = monitor.get_snapshot(step, loss.item(), lr)

    Integration with TensorBoard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter('runs/experiment')
        monitor = TrainingMonitor(tensorboard_writer=writer)

        # Monitor will automatically log to TensorBoard
    """

    # Thresholds for health checks
    GRADIENT_NORM_MIN = 1e-7
    GRADIENT_NORM_MAX = 100.0
    ZERO_RATIO_MAX = 0.9  # Max 90% zeros
    UPDATE_RATIO_MIN = 0.001
    UPDATE_RATIO_MAX = 0.1
    SATURATION_MAX = 0.5  # Max 50% saturated activations

    def __init__(
        self,
        tensorboard_writer: Optional[Any] = None,
        wandb_run: Optional[Any] = None,
        log_frequency: int = 100,
        track_activations: bool = True,
        track_weights: bool = True,
    ):
        """
        Initialize training monitor.

        Args:
            tensorboard_writer: Optional TensorBoard SummaryWriter
            wandb_run: Optional wandb run object
            log_frequency: How often to log histograms (in steps)
            track_activations: Whether to track activation distributions
            track_weights: Whether to track weight distributions
        """
        self.writer = tensorboard_writer
        self.wandb = wandb_run
        self.log_frequency = log_frequency
        self.track_activations = track_activations
        self.track_weights = track_weights

        # State
        self._step = 0
        self._snapshots: List[TrainingSnapshot] = []
        self._previous_weights: Dict[str, np.ndarray] = {}
        self._activation_buffers: Dict[str, np.ndarray] = {}

    def record_gradients(
        self,
        named_parameters: List[Tuple[str, np.ndarray]],
        step: Optional[int] = None,
    ) -> List[GradientReport]:
        """
        Record gradient statistics for model parameters.

        Args:
            named_parameters: List of (name, gradient) tuples
            step: Optional step number (uses internal counter if not provided)

        Returns:
            List of GradientReport for each parameter
        """
        if step is not None:
            self._step = step

        reports = []
        should_log = self._step % self.log_frequency == 0

        for name, grad in named_parameters:
            if grad is None:
                continue

            grad = _ensure_array(grad)

            # Compute statistics
            report = GradientReport(
                layer_name=name,
                mean=float(np.mean(grad)),
                std=float(np.std(grad)),
                min_val=float(np.min(grad)),
                max_val=float(np.max(grad)),
                norm=float(np.linalg.norm(grad)),
                zero_ratio=float(np.mean(np.abs(grad) < 1e-10)),
                nan_count=int(np.sum(np.isnan(grad))),
                inf_count=int(np.sum(np.isinf(grad))),
            )
            reports.append(report)

            # Log to TensorBoard
            if should_log and self.writer is not None:
                self._log_gradient_to_tensorboard(name, grad, report)

            # Log to wandb
            if should_log and self.wandb is not None:
                self._log_gradient_to_wandb(name, grad, report)

        return reports

    def record_activations(
        self,
        layer_name: str,
        activations: np.ndarray,
        step: Optional[int] = None,
        activation_type: str = "relu",
    ) -> ActivationReport:
        """
        Record activation statistics for a layer.

        Args:
            layer_name: Name of the layer
            activations: Activation tensor
            step: Optional step number
            activation_type: Type of activation ('relu', 'sigmoid', 'tanh', 'gelu')

        Returns:
            ActivationReport for this layer
        """
        if step is not None:
            self._step = step

        activations = _ensure_array(activations)

        # Compute saturation ratio based on activation type
        if activation_type == "sigmoid":
            # Saturated if > 0.95 or < 0.05
            saturation = float(
                np.mean((activations > 0.95) | (activations < 0.05))
            )
        elif activation_type == "tanh":
            # Saturated if > 0.95 or < -0.95
            saturation = float(
                np.mean((activations > 0.95) | (activations < -0.95))
            )
        else:
            # For ReLU/GELU: use zero ratio
            saturation = 0.0

        report = ActivationReport(
            layer_name=layer_name,
            mean=float(np.mean(activations)),
            std=float(np.std(activations)),
            min_val=float(np.min(activations)),
            max_val=float(np.max(activations)),
            zero_ratio=float(np.mean(np.abs(activations) < 1e-10)),
            saturation_ratio=saturation,
            nan_count=int(np.sum(np.isnan(activations))),
        )

        # Store for later analysis
        self._activation_buffers[layer_name] = activations

        # Log to TensorBoard
        should_log = self._step % self.log_frequency == 0
        if should_log and self.writer is not None:
            self._log_activation_to_tensorboard(layer_name, activations, report)

        return report

    def record_weights(
        self,
        named_parameters: List[Tuple[str, np.ndarray]],
        step: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Record weight statistics for model parameters.

        Args:
            named_parameters: List of (name, weight) tuples
            step: Optional step number

        Returns:
            Dictionary of weight statistics per layer
        """
        if step is not None:
            self._step = step

        stats = {}
        should_log = self._step % self.log_frequency == 0

        for name, weight in named_parameters:
            weight = _ensure_array(weight)

            layer_stats = {
                "mean": float(np.mean(weight)),
                "std": float(np.std(weight)),
                "min": float(np.min(weight)),
                "max": float(np.max(weight)),
                "norm": float(np.linalg.norm(weight)),
                "nan_count": int(np.sum(np.isnan(weight))),
            }
            stats[name] = layer_stats

            # Log to TensorBoard
            if should_log and self.writer is not None:
                self._log_weight_to_tensorboard(name, weight)

        return stats

    def compute_weight_update_ratio(
        self,
        layer_name: str,
        current_weight: np.ndarray,
        previous_weight: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float,
    ) -> WeightUpdateReport:
        """
        Compute weight update ratio for a layer.

        The update ratio is ||weight_update|| / ||weight||.
        This should typically be between 0.001 and 0.1.

        Args:
            layer_name: Name of the layer
            current_weight: Weight after update
            previous_weight: Weight before update
            gradient: Gradient used for update
            learning_rate: Learning rate

        Returns:
            WeightUpdateReport with update statistics
        """
        current_weight = _ensure_array(current_weight)
        previous_weight = _ensure_array(previous_weight)
        gradient = _ensure_array(gradient)

        update = current_weight - previous_weight
        weight_norm = float(np.linalg.norm(current_weight))
        update_norm = float(np.linalg.norm(update))
        gradient_norm = float(np.linalg.norm(gradient))

        # Avoid division by zero
        update_ratio = update_norm / (weight_norm + 1e-10)

        return WeightUpdateReport(
            layer_name=layer_name,
            weight_norm=weight_norm,
            update_norm=update_norm,
            update_ratio=update_ratio,
            gradient_norm=gradient_norm,
        )

    def check_health(
        self,
        gradient_reports: List[GradientReport],
        activation_reports: Optional[List[ActivationReport]] = None,
        weight_update_reports: Optional[List[WeightUpdateReport]] = None,
    ) -> Tuple[MonitorStatus, List[str]]:
        """
        Check training health based on collected reports.

        Args:
            gradient_reports: Gradient statistics
            activation_reports: Optional activation statistics
            weight_update_reports: Optional weight update statistics

        Returns:
            Tuple of (status, warnings)
        """
        warnings = []
        has_critical = False
        has_warning = False

        # Check gradients
        for report in gradient_reports:
            # Check for NaN/Inf
            if report.nan_count > 0:
                warnings.append(
                    f"[CRITICAL] {report.layer_name}: {report.nan_count} NaN values in gradient"
                )
                has_critical = True

            if report.inf_count > 0:
                warnings.append(
                    f"[CRITICAL] {report.layer_name}: {report.inf_count} Inf values in gradient"
                )
                has_critical = True

            # Check gradient norm
            if report.norm < self.GRADIENT_NORM_MIN:
                warnings.append(
                    f"[WARNING] {report.layer_name}: Vanishing gradient (norm={report.norm:.2e})"
                )
                has_warning = True

            if report.norm > self.GRADIENT_NORM_MAX:
                warnings.append(
                    f"[WARNING] {report.layer_name}: Exploding gradient (norm={report.norm:.2e})"
                )
                has_warning = True

            # Check zero ratio
            if report.zero_ratio > self.ZERO_RATIO_MAX:
                warnings.append(
                    f"[WARNING] {report.layer_name}: High zero ratio ({report.zero_ratio:.1%})"
                )
                has_warning = True

        # Check activations
        if activation_reports:
            for report in activation_reports:
                if report.nan_count > 0:
                    warnings.append(
                        f"[CRITICAL] {report.layer_name}: {report.nan_count} NaN values in activation"
                    )
                    has_critical = True

                if report.saturation_ratio > self.SATURATION_MAX:
                    warnings.append(
                        f"[WARNING] {report.layer_name}: High saturation ({report.saturation_ratio:.1%})"
                    )
                    has_warning = True

        # Check weight updates
        if weight_update_reports:
            for report in weight_update_reports:
                if report.update_ratio < self.UPDATE_RATIO_MIN:
                    warnings.append(
                        f"[WARNING] {report.layer_name}: Low update ratio ({report.update_ratio:.2e})"
                    )
                    has_warning = True

                if report.update_ratio > self.UPDATE_RATIO_MAX:
                    warnings.append(
                        f"[WARNING] {report.layer_name}: High update ratio ({report.update_ratio:.2e})"
                    )
                    has_warning = True

        # Determine status
        if has_critical:
            status = MonitorStatus.CRITICAL
        elif has_warning:
            status = MonitorStatus.WARNING
        else:
            status = MonitorStatus.HEALTHY

        return status, warnings

    def get_snapshot(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_reports: Optional[List[GradientReport]] = None,
        activation_reports: Optional[List[ActivationReport]] = None,
        weight_update_reports: Optional[List[WeightUpdateReport]] = None,
    ) -> TrainingSnapshot:
        """
        Create a training snapshot for logging.

        Args:
            step: Current training step
            loss: Current loss value
            learning_rate: Current learning rate
            gradient_reports: Optional gradient reports
            activation_reports: Optional activation reports
            weight_update_reports: Optional weight update reports

        Returns:
            TrainingSnapshot with current state
        """
        self._step = step

        gradient_reports = gradient_reports or []
        activation_reports = activation_reports or []
        weight_update_reports = weight_update_reports or []

        status, warnings = self.check_health(
            gradient_reports, activation_reports, weight_update_reports
        )

        snapshot = TrainingSnapshot(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            gradient_reports=gradient_reports,
            activation_reports=activation_reports,
            weight_update_reports=weight_update_reports,
            status=status,
            warnings=warnings,
        )

        self._snapshots.append(snapshot)
        return snapshot

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value to TensorBoard and/or wandb.

        Args:
            tag: Tag for the metric
            value: Scalar value
            step: Optional step number
        """
        if step is not None:
            self._step = step

        if self.writer is not None:
            self.writer.add_scalar(tag, value, self._step)

        if self.wandb is not None:
            self.wandb.log({tag: value}, step=self._step)

    def log_histogram(
        self, tag: str, values: np.ndarray, step: Optional[int] = None
    ):
        """
        Log a histogram to TensorBoard and/or wandb.

        Args:
            tag: Tag for the histogram
            values: Values to histogram
            step: Optional step number
        """
        if step is not None:
            self._step = step

        if self.writer is not None:
            self.writer.add_histogram(tag, values, self._step)

        if self.wandb is not None:
            import wandb as wandb_lib

            self.wandb.log({tag: wandb_lib.Histogram(values)}, step=self._step)

    def _log_gradient_to_tensorboard(
        self, name: str, grad: np.ndarray, report: GradientReport
    ):
        """Log gradient to TensorBoard."""
        # Sanitize name for TensorBoard
        safe_name = name.replace(".", "/")

        # Log histogram
        self.writer.add_histogram(f"Gradients/{safe_name}", grad, self._step)

        # Log scalar stats
        self.writer.add_scalar(f"GradientNorm/{safe_name}", report.norm, self._step)
        self.writer.add_scalar(f"GradientMean/{safe_name}", report.mean, self._step)
        self.writer.add_scalar(f"GradientStd/{safe_name}", report.std, self._step)
        self.writer.add_scalar(f"GradientZeroRatio/{safe_name}", report.zero_ratio, self._step)

    def _log_gradient_to_wandb(
        self, name: str, grad: np.ndarray, report: GradientReport
    ):
        """Log gradient to wandb."""
        import wandb as wandb_lib

        safe_name = name.replace(".", "/")

        self.wandb.log(
            {
                f"Gradients/{safe_name}": wandb_lib.Histogram(grad),
                f"GradientNorm/{safe_name}": report.norm,
                f"GradientMean/{safe_name}": report.mean,
                f"GradientZeroRatio/{safe_name}": report.zero_ratio,
            },
            step=self._step,
        )

    def _log_activation_to_tensorboard(
        self, name: str, activations: np.ndarray, report: ActivationReport
    ):
        """Log activation to TensorBoard."""
        safe_name = name.replace(".", "/")

        self.writer.add_histogram(f"Activations/{safe_name}", activations, self._step)
        self.writer.add_scalar(f"ActivationMean/{safe_name}", report.mean, self._step)
        self.writer.add_scalar(f"ActivationStd/{safe_name}", report.std, self._step)
        self.writer.add_scalar(f"ActivationZeroRatio/{safe_name}", report.zero_ratio, self._step)

        if report.saturation_ratio > 0:
            self.writer.add_scalar(
                f"ActivationSaturation/{safe_name}", report.saturation_ratio, self._step
            )

    def _log_weight_to_tensorboard(self, name: str, weight: np.ndarray):
        """Log weight to TensorBoard."""
        safe_name = name.replace(".", "/")
        self.writer.add_histogram(f"Weights/{safe_name}", weight, self._step)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected data.

        Returns:
            Dictionary with training summary
        """
        if not self._snapshots:
            return {"error": "No snapshots collected"}

        # Get latest snapshot
        latest = self._snapshots[-1]

        # Compute aggregate stats
        losses = [s.loss for s in self._snapshots]
        statuses = [s.status.value for s in self._snapshots]

        return {
            "total_steps": len(self._snapshots),
            "latest_step": latest.step,
            "latest_loss": latest.loss,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "latest_status": latest.status.value,
            "total_warnings": sum(len(s.warnings) for s in self._snapshots),
            "status_distribution": {
                "healthy": statuses.count("healthy"),
                "warning": statuses.count("warning"),
                "critical": statuses.count("critical"),
            },
        }


# =============================================================================
# Utility Functions
# =============================================================================


def compute_gradient_histogram(
    gradients: List[np.ndarray], bins: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute histogram of all gradients combined.

    Args:
        gradients: List of gradient arrays
        bins: Number of histogram bins

    Returns:
        Dictionary with histogram data
    """
    all_grads = np.concatenate([g.flatten() for g in gradients])
    hist, bin_edges = np.histogram(all_grads, bins=bins)

    return {
        "counts": hist,
        "bin_edges": bin_edges,
        "mean": float(np.mean(all_grads)),
        "std": float(np.std(all_grads)),
        "min": float(np.min(all_grads)),
        "max": float(np.max(all_grads)),
    }


def detect_dead_neurons(
    activation: np.ndarray, threshold: float = 0.99
) -> Tuple[bool, float]:
    """
    Detect if a layer has dead neurons.

    A neuron is "dead" if it outputs zero for almost all inputs.

    Args:
        activation: Activation tensor (batch, features, ...)
        threshold: Ratio of zeros to consider neuron dead

    Returns:
        Tuple of (has_dead_neurons, dead_ratio)
    """
    # Flatten batch and spatial dimensions
    if activation.ndim > 2:
        activation = activation.reshape(activation.shape[0], -1)

    # Compute zero ratio per neuron (across batch)
    zero_ratio = np.mean(activation == 0, axis=0)

    # Find dead neurons
    dead_count = np.sum(zero_ratio > threshold)
    dead_ratio = dead_count / len(zero_ratio)

    return dead_ratio > 0.1, float(dead_ratio)


def compute_activation_distribution(
    activation: np.ndarray, activation_type: str = "relu"
) -> Dict[str, float]:
    """
    Compute distribution statistics for activations.

    Args:
        activation: Activation tensor
        activation_type: Type of activation function

    Returns:
        Dictionary with distribution statistics
    """
    activation = _ensure_array(activation)
    flat = activation.flatten()

    stats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "zero_ratio": float(np.mean(np.abs(flat) < 1e-10)),
    }

    if activation_type == "relu":
        # For ReLU, also track positive ratio
        stats["positive_ratio"] = float(np.mean(flat > 0))
    elif activation_type == "sigmoid":
        # For sigmoid, track saturation
        stats["saturation_low"] = float(np.mean(flat < 0.1))
        stats["saturation_high"] = float(np.mean(flat > 0.9))
    elif activation_type == "tanh":
        # For tanh, track saturation
        stats["saturation_low"] = float(np.mean(flat < -0.9))
        stats["saturation_high"] = float(np.mean(flat > 0.9))

    return stats


# =============================================================================
# Module Constants
# =============================================================================


TRAINING_MONITOR_COMPONENTS = [
    "TrainingMonitor",
    "GradientReport",
    "ActivationReport",
    "WeightUpdateReport",
    "TrainingSnapshot",
    "MonitorStatus",
    "compute_gradient_histogram",
    "detect_dead_neurons",
    "compute_activation_distribution",
]
