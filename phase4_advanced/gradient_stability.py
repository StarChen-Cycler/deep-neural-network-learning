"""
Gradient Stability: Diagnostics and Solutions for Vanishing/Exploding Gradients.

This module provides:
    - Gradient clipping (clip_grad_norm, clip_grad_value)
    - Gradient flow diagnostics and analysis
    - Solutions for gradient problems (skip connections, layer scale)

Theory:
    Vanishing Gradient Problem:
        - Gradients become exponentially smaller in deep networks
        - Causes: Sigmoid/Tanh activations, poor initialization
        - Solutions: ReLU/GELU, BatchNorm, Skip connections, proper init

    Exploding Gradient Problem:
        - Gradients become exponentially larger
        - Causes: Deep RNNs, high learning rate, poor initialization
        - Solutions: Gradient clipping, lower LR, Adam optimizer

    Gradient Clipping:
        - clip_grad_norm: clips gradient norm to max_norm
            g_clipped = g * min(1, max_norm / ||g||)
        - clip_grad_value: clips gradient values to [-clip_value, clip_value]

    Skip Connections (ResNet):
        output = F(x) + x
        dL/dx = dL/d(output) * (dF/dx + 1)
        The "+1" ensures gradient flow regardless of F

References:
    - On the difficulty of training recurrent neural networks (Pascanu et al., 2013)
    - Deep Residual Learning for Image Recognition (He et al., 2015)
    - Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
"""

from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
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
# Gradient Clipping Functions
# =============================================================================


def clip_grad_norm(
    gradients: List[np.ndarray],
    max_norm: float,
    norm_type: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """
    Clip gradients by global norm.

    Clips the norm of the gradients to max_norm using the specified norm type.
    This is the standard approach for preventing exploding gradients.

    Formula:
        total_norm = ||grads||_{norm_type}
        if total_norm > max_norm:
            scale = max_norm / total_norm
            clipped_grads = [g * scale for g in grads]
        else:
            clipped_grads = grads

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed norm
        norm_type: Norm type (2.0 for L2, 1.0 for L1, float('inf') for max norm)

    Returns:
        Tuple of (clipped_gradients, total_norm)

    Example:
        >>> grads = [np.array([3.0, 4.0]), np.array([1.0, 2.0])]
        >>> clipped, norm = clip_grad_norm(grads, max_norm=1.0)
        >>> print(f"Original norm: {norm:.4f}")  # 5.4772
        >>> print(f"Clipped norm: {np.sqrt(sum(np.sum(g**2) for g in clipped)):.4f}")  # 1.0

    References:
        - On the difficulty of training recurrent neural networks (Pascanu et al., 2013)
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    # Ensure all gradients are arrays
    grads = [_ensure_array(g) for g in gradients]

    # Compute total norm
    if norm_type == float("inf"):
        # Max norm
        total_norm = max(np.max(np.abs(g)) for g in grads if g.size > 0)
    else:
        # p-norm
        total_norm = sum(np.sum(np.abs(g) ** norm_type) for g in grads)
        total_norm = total_norm ** (1.0 / norm_type)

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        # Need to clip
        clipped_grads = [g * clip_coef for g in grads]
    else:
        # No clipping needed
        clipped_grads = [g.copy() for g in grads]

    return clipped_grads, total_norm


def clip_grad_value(
    gradients: List[np.ndarray],
    clip_value: float,
) -> List[np.ndarray]:
    """
    Clip gradients by value.

    Clips gradient values to [-clip_value, clip_value].
    This is simpler than norm clipping but less principled.

    Formula:
        clipped_grad = clip(grad, -clip_value, clip_value)

    Args:
        gradients: List of gradient arrays
        clip_value: Maximum absolute value allowed

    Returns:
        Clipped gradients

    Example:
        >>> grads = [np.array([5.0, -3.0, 2.0])]
        >>> clipped = clip_grad_value(grads, clip_value=2.0)
        >>> print(clipped[0])  # [2.0, -2.0, 2.0]
    """
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")

    grads = [_ensure_array(g) for g in gradients]
    return [np.clip(g, -clip_value, clip_value) for g in grads]


# =============================================================================
# Gradient Statistics and Diagnostics
# =============================================================================


@dataclass
class GradientStats:
    """
    Statistics for a single gradient tensor.

    Attributes:
        mean: Mean value of gradient
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        norm_l2: L2 norm
        norm_l1: L1 norm
        zeros_ratio: Fraction of near-zero values (< 1e-7)
        nan_count: Number of NaN values
        inf_count: Number of Inf values
    """

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    norm_l2: float = 0.0
    norm_l1: float = 0.0
    zeros_ratio: float = 0.0
    nan_count: int = 0
    inf_count: int = 0

    @classmethod
    def from_tensor(cls, grad: np.ndarray, eps: float = 1e-7) -> "GradientStats":
        """Compute statistics from a gradient tensor."""
        grad = _ensure_array(grad)

        if grad.size == 0:
            return cls()

        flat = grad.flatten()
        abs_flat = np.abs(flat)

        return cls(
            mean=float(np.mean(flat)),
            std=float(np.std(flat)),
            min=float(np.min(flat)),
            max=float(np.max(flat)),
            norm_l2=float(np.sqrt(np.sum(flat**2))),
            norm_l1=float(np.sum(abs_flat)),
            zeros_ratio=float(np.mean(abs_flat < eps)),
            nan_count=int(np.sum(np.isnan(flat))),
            inf_count=int(np.sum(np.isinf(flat))),
        )

    def is_healthy(self) -> bool:
        """Check if gradient is healthy (no NaN, no Inf, not all zeros)."""
        return (
            self.nan_count == 0
            and self.inf_count == 0
            and self.zeros_ratio < 0.99
            and not np.isnan(self.norm_l2)
            and not np.isinf(self.norm_l2)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "norm_l2": self.norm_l2,
            "norm_l1": self.norm_l1,
            "zeros_ratio": self.zeros_ratio,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "is_healthy": self.is_healthy(),
        }


class GradientFlowAnalyzer:
    """
    Analyzer for gradient flow through network layers.

    Tracks gradient statistics at each layer to diagnose:
        - Vanishing gradients (norm decreasing through layers)
        - Exploding gradients (norm increasing through layers)
        - Dead neurons (high zeros_ratio)

    Usage:
        analyzer = GradientFlowAnalyzer(num_layers=10)

        # During backward pass
        for i, grad in enumerate(layer_grads):
            analyzer.record_layer_gradient(i, grad)

        # Get diagnosis
        report = analyzer.get_flow_report()
    """

    def __init__(self, num_layers: int):
        """
        Initialize analyzer.

        Args:
            num_layers: Number of layers to track
        """
        self.num_layers = num_layers
        self.layer_stats: List[GradientStats] = []
        self._recorded = False

    def record_layer_gradient(self, layer_idx: int, grad: np.ndarray) -> GradientStats:
        """
        Record gradient statistics for a layer.

        Args:
            layer_idx: Layer index (0 = closest to loss)
            grad: Gradient tensor for this layer

        Returns:
            GradientStats for this gradient
        """
        stats = GradientStats.from_tensor(grad)

        # Extend list if needed
        while len(self.layer_stats) <= layer_idx:
            self.layer_stats.append(None)

        self.layer_stats[layer_idx] = stats
        self._recorded = True
        return stats

    def record_gradients(self, gradients: List[np.ndarray]) -> List[GradientStats]:
        """
        Record gradients for all layers at once.

        Args:
            gradients: List of gradients (ordered from output to input)

        Returns:
            List of GradientStats
        """
        self.layer_stats = []
        for i, grad in enumerate(gradients):
            stats = GradientStats.from_tensor(grad)
            self.layer_stats.append(stats)
        self._recorded = True
        return self.layer_stats

    def get_flow_report(self) -> Dict[str, Any]:
        """
        Generate gradient flow report.

        Returns:
            Dictionary with flow analysis including:
                - per_layer_stats: Stats for each layer
                - flow_ratio: Ratio of each layer norm to first layer
                - diagnosis: 'healthy', 'vanishing', 'exploding', or 'unstable'
        """
        if not self._recorded:
            return {"error": "No gradients recorded"}

        # Compute flow ratios
        norms = [s.norm_l2 for s in self.layer_stats if s is not None]
        if not norms or norms[0] == 0:
            return {"error": "Invalid gradient norms"}

        first_norm = norms[0]
        flow_ratios = [n / (first_norm + 1e-10) for n in norms]

        # Detect vanishing: last layer norm < 0.01 * first layer
        is_vanishing = flow_ratios[-1] < 0.01 if flow_ratios else False

        # Detect exploding: any layer norm > 100 * first layer
        is_exploding = any(r > 100 for r in flow_ratios)

        # Check for NaN/Inf
        has_nan = any(s.nan_count > 0 for s in self.layer_stats if s)
        has_inf = any(s.inf_count > 0 for s in self.layer_stats if s)

        # Determine diagnosis
        if has_nan or has_inf:
            diagnosis = "unstable"
        elif is_exploding:
            diagnosis = "exploding"
        elif is_vanishing:
            diagnosis = "vanishing"
        else:
            diagnosis = "healthy"

        return {
            "per_layer_stats": [s.to_dict() if s else None for s in self.layer_stats],
            "flow_ratios": flow_ratios,
            "norms": norms,
            "is_vanishing": is_vanishing,
            "is_exploding": is_exploding,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "diagnosis": diagnosis,
        }


def detect_vanishing_gradient(
    gradients: List[np.ndarray],
    threshold_ratio: float = 0.01,
) -> Tuple[bool, float]:
    """
    Detect vanishing gradient problem.

    Checks if the gradient norm decreases by more than threshold_ratio
    from the first layer to the last layer.

    Args:
        gradients: List of gradients (ordered from output to input)
        threshold_ratio: Minimum allowed ratio of last/first norm

    Returns:
        Tuple of (is_vanishing, ratio)
    """
    grads = [_ensure_array(g) for g in gradients]

    norms = [np.sqrt(np.sum(g**2)) for g in grads if g.size > 0]
    if len(norms) < 2:
        return False, 1.0

    ratio = norms[-1] / (norms[0] + 1e-10)
    return ratio < threshold_ratio, ratio


def detect_exploding_gradient(
    gradients: List[np.ndarray],
    threshold_ratio: float = 100.0,
) -> Tuple[bool, float]:
    """
    Detect exploding gradient problem.

    Checks if any gradient norm is more than threshold_ratio times
    the first layer norm.

    Args:
        gradients: List of gradients (ordered from output to input)
        threshold_ratio: Maximum allowed ratio of any norm to first norm

    Returns:
        Tuple of (is_exploding, max_ratio)
    """
    grads = [_ensure_array(g) for g in gradients]

    norms = [np.sqrt(np.sum(g**2)) for g in grads if g.size > 0]
    if len(norms) < 2:
        return False, 1.0

    first_norm = norms[0] + 1e-10
    ratios = [n / first_norm for n in norms]
    max_ratio = max(ratios)
    return max_ratio > threshold_ratio, max_ratio


# =============================================================================
# Solutions for Gradient Problems
# =============================================================================


def apply_skip_connection(
    x: np.ndarray,
    fx: np.ndarray,
    mode: str = "add",
    projection: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply skip connection (residual connection).

    Skip connections allow gradients to flow directly through the network,
    solving the vanishing gradient problem.

    Formulas:
        Add: output = F(x) + x
        Concat: output = concat(F(x), x)

    Args:
        x: Input tensor
        fx: Output of residual block F(x)
        mode: 'add' for addition, 'concat' for concatenation
        projection: Optional projection matrix for dimension mismatch

    Returns:
        Output tensor with skip connection

    Example:
        >>> x = np.random.randn(32, 64)
        >>> fx = np.random.randn(32, 64)  # F(x)
        >>> output = apply_skip_connection(x, fx, mode='add')
        >>> output.shape  # (32, 64)
    """
    x = _ensure_array(x)
    fx = _ensure_array(fx)

    if mode == "add":
        if projection is not None:
            x = x @ projection
        return fx + x
    elif mode == "concat":
        return np.concatenate([fx, x], axis=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'add' or 'concat'.")


class LayerScale:
    """
    Layer Scale for stabilizing very deep networks.

    Multiplies the output of a layer by a learnable scalar per channel,
    initialized to a small value (e.g., 1e-5).

    Formula:
        output = gamma * x
        where gamma is initialized to initial_value

    This helps with training very deep networks (100+ layers) by
    allowing the network to start in identity-like behavior.

    References:
        - Going deeper with Image Transformers (Touvron et al., 2021)
    """

    def __init__(self, dim: int, initial_value: float = 1e-5):
        """
        Initialize LayerScale.

        Args:
            dim: Dimension of the input/output
            initial_value: Initial value for the scale parameter
        """
        self.dim = dim
        self.gamma = np.ones(dim) * initial_value
        self.grad_gamma: Optional[np.ndarray] = None
        self._cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., dim)

        Returns:
            Scaled tensor
        """
        x = _ensure_array(x)
        self._cache = x
        return x * self.gamma

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        grad_output = _ensure_array(grad_output)

        # Gradient w.r.t. gamma
        self.grad_gamma = np.sum(grad_output * self._cache, axis=tuple(range(self._cache.ndim - 1)))

        # Gradient w.r.t. input
        grad_input = grad_output * self.gamma
        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return parameters."""
        return [self.gamma]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        return [self.grad_gamma]


# =============================================================================
# Utility Functions for Deep Networks
# =============================================================================


def compute_gradient_norm(
    parameters: List[np.ndarray],
    norm_type: float = 2.0,
) -> float:
    """
    Compute total gradient norm.

    Args:
        parameters: List of parameter arrays with .grad attribute or list of gradients
        norm_type: Norm type (2.0 for L2)

    Returns:
        Total norm
    """
    if norm_type == float("inf"):
        return float(max(np.max(np.abs(p)) for p in parameters if p.size > 0))
    else:
        total = sum(np.sum(np.abs(p) ** norm_type) for p in parameters)
        return float(total ** (1.0 / norm_type))


def get_activation_gradient_scale(activation_name: str) -> float:
    """
    Get typical gradient scale factor for activations.

    Different activations have different gradient scales:
        - ReLU: 0.5 (half the time zero)
        - Sigmoid: 0.25 (maximum at x=0)
        - Tanh: 1.0 (maximum at x=0)
        - GELU: ~0.4 (smooth approximation of ReLU)

    Args:
        activation_name: Name of activation function

    Returns:
        Typical gradient scale factor
    """
    scales = {
        "relu": 0.5,
        "leaky_relu": 0.5,
        "sigmoid": 0.25,
        "tanh": 1.0,
        "gelu": 0.4,
        "swish": 0.5,
        "identity": 1.0,
    }
    return scales.get(activation_name.lower(), 1.0)


# =============================================================================
# Registry
# =============================================================================

GRADIENT_CLIPPING_FUNCTIONS = {
    "norm": clip_grad_norm,
    "value": clip_grad_value,
}


def get_gradient_clipper(name: str) -> Callable:
    """Get gradient clipping function by name."""
    if name not in GRADIENT_CLIPPING_FUNCTIONS:
        available = list(GRADIENT_CLIPPING_FUNCTIONS.keys())
        raise ValueError(f"Unknown clipper: {name}. Available: {available}")
    return GRADIENT_CLIPPING_FUNCTIONS[name]
