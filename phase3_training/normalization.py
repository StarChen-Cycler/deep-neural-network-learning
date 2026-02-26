"""
Normalization techniques for deep neural networks.

This module provides implementations of 4 normalization methods:
    - BatchNorm: Normalizes across batch dimension (for CNNs)
    - LayerNorm: Normalizes across feature dimension (for Transformers)
    - InstanceNorm: Normalizes per sample per channel (for style transfer)
    - GroupNorm: Normalizes across channel groups (batch-size independent)

Each normalization follows the formula:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

Key differences:
    - BatchNorm: mean/var computed over (N, H, W) for each channel
    - LayerNorm: mean/var computed over (C,) or specified dims for each sample
    - InstanceNorm: mean/var computed over (H, W) for each sample and channel
    - GroupNorm: mean/var computed over (C//G, H, W) for each sample and group

References:
    - Batch Normalization: https://arxiv.org/abs/1502.03167
    - Layer Normalization: https://arxiv.org/abs/1607.06450
    - Instance Normalization: https://arxiv.org/abs/1607.08022
    - Group Normalization: https://arxiv.org/abs/1803.08494
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np

ArrayLike = Union[np.ndarray, float, int]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array if not already."""
    return np.asarray(x, dtype=np.float64)


# =============================================================================
# BatchNorm1d - For 1D inputs (N, C) or (N, C, L)
# =============================================================================


class BatchNorm1d:
    """
    Batch Normalization for 1D inputs.

    For input (N, C) or (N, C, L):
        Normalizes across batch dimension (axis 0 for (N,C), axes (0,2) for (N,C,L))

    Formula:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    Attributes:
        gamma: Scale parameter (C,)
        beta: Shift parameter (C,)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        momentum: Momentum for running stats update
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """
        Initialize BatchNorm1d.

        Args:
            num_features: Number of features (C)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma: np.ndarray = np.ones(num_features, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(num_features, dtype=np.float64)

        # Running statistics (for inference)
        self.running_mean: np.ndarray = np.zeros(num_features, dtype=np.float64)
        self.running_var: np.ndarray = np.ones(num_features, dtype=np.float64)

        # Training mode flag
        self.training: bool = True

        # Cache for backward
        self._cache: Optional[Dict[str, np.ndarray]] = None

        # Gradients
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for 1D batch normalization.

        Args:
            x: Input tensor of shape (N, C) or (N, C, L)

        Returns:
            Normalized tensor of same shape
        """
        x = _ensure_array(x)

        if x.ndim == 2:
            # (N, C) case
            N, C = x.shape

            if self.training:
                mean = np.mean(x, axis=0, keepdims=True)
                var = np.var(x, axis=0, keepdims=True)

                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            else:
                mean = self.running_mean.reshape(1, -1)
                var = self.running_var.reshape(1, -1)

            std = np.sqrt(var + self.eps)
            x_norm = (x - mean) / std
            out = self.gamma * x_norm + self.beta

            self._cache = {"x": x, "x_norm": x_norm, "mean": mean, "std": std, "N": N}

        else:
            # (N, C, L) case
            N, C, L = x.shape

            if self.training:
                mean = np.mean(x, axis=(0, 2), keepdims=True)
                var = np.var(x, axis=(0, 2), keepdims=True)

                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            else:
                mean = self.running_mean.reshape(1, -1, 1)
                var = self.running_var.reshape(1, -1, 1)

            std = np.sqrt(var + self.eps)
            x_norm = (x - mean) / std
            out = self.gamma.reshape(1, -1, 1) * x_norm + self.beta.reshape(1, -1, 1)

            self._cache = {"x": x, "x_norm": x_norm, "mean": mean, "std": std, "N": N * L}

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        N = self._cache["N"]

        if x.ndim == 2:
            # Gradient w.r.t. gamma and beta
            self.grad_gamma = np.sum(grad_output * x_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)

            # Gradient w.r.t. x_norm
            dx_norm = grad_output * self.gamma

            # Gradient w.r.t. variance
            dvar = np.sum(dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3), axis=0, keepdims=True)

            # Gradient w.r.t. mean
            dmean = np.sum(dx_norm * (-1 / std), axis=0, keepdims=True) + dvar * np.mean(
                -2 * (x - self._cache["mean"]), axis=0, keepdims=True
            )

            # Gradient w.r.t. x
            grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / N + dmean / N

        else:
            # (N, C, L) case
            # Gradient w.r.t. gamma and beta
            self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2))
            self.grad_beta = np.sum(grad_output, axis=(0, 2))

            # Gradient w.r.t. x_norm
            dx_norm = grad_output * self.gamma.reshape(1, -1, 1)

            # Gradient w.r.t. variance
            dvar = np.sum(
                dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3),
                axis=(0, 2),
                keepdims=True,
            )

            # Gradient w.r.t. mean
            dmean = np.sum(dx_norm * (-1 / std), axis=(0, 2), keepdims=True) + dvar * np.mean(
                -2 * (x - self._cache["mean"]), axis=(0, 2), keepdims=True
            )

            # Gradient w.r.t. x
            grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / N + dmean / N

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        return [self.gamma, self.beta]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        return [self.grad_gamma, self.grad_beta]

    def zero_grad(self) -> None:
        """Reset gradients."""
        self.grad_gamma = None
        self.grad_beta = None

    def eval(self) -> None:
        """Set to evaluation mode."""
        self.training = False

    def train(self) -> None:
        """Set to training mode."""
        self.training = True


# =============================================================================
# BatchNorm2d - For 2D inputs (N, C, H, W)
# =============================================================================


class BatchNorm2d:
    """
    Batch Normalization for 2D inputs (N, C, H, W).

    Normalizes across batch and spatial dimensions:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    During training: uses batch statistics
    During inference: uses running statistics

    Attributes:
        gamma: Scale parameter (C,)
        beta: Shift parameter (C,)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        momentum: Momentum for running stats update
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """
        Initialize BatchNorm2d.

        Args:
            num_features: Number of channels (C)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma: np.ndarray = np.ones(num_features, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(num_features, dtype=np.float64)

        # Running statistics (for inference)
        self.running_mean: np.ndarray = np.zeros(num_features, dtype=np.float64)
        self.running_var: np.ndarray = np.ones(num_features, dtype=np.float64)

        # Training mode flag
        self.training: bool = True

        # Cache for backward
        self._cache: Optional[Dict[str, np.ndarray]] = None

        # Gradients
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Normalized tensor of same shape
        """
        x = _ensure_array(x)
        batch, channels, height, width = x.shape

        if self.training:
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # Use running statistics for inference
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std

        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

        # Cache for backward
        self._cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "std": std,
            "N": batch * height * width,
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        N = self._cache["N"]

        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2, 3))
        self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))

        # Gradient w.r.t. x_norm
        dx_norm = grad_output * self.gamma.reshape(1, -1, 1, 1)

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3),
            axis=(0, 2, 3),
            keepdims=True,
        )

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * (-1 / std), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(
            -2 * (x - self._cache["mean"]), axis=(0, 2, 3), keepdims=True
        )

        # Gradient w.r.t. x
        grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / N + dmean / N

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        return [self.gamma, self.beta]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        return [self.grad_gamma, self.grad_beta]

    def zero_grad(self) -> None:
        """Reset gradients."""
        self.grad_gamma = None
        self.grad_beta = None

    def eval(self) -> None:
        """Set to evaluation mode."""
        self.training = False

    def train(self) -> None:
        """Set to training mode."""
        self.training = True


# =============================================================================
# LayerNorm - Normalize across feature dimension
# =============================================================================


class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the last dimension(s), independent of batch size.
    Commonly used in Transformers and NLP models.

    Formula:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    For input (N, D) or (N, L, D):
        mean/var computed over D (or last normalized_shape dims)

    Attributes:
        normalized_shape: Shape over which to normalize (typically D or (L, D))
        gamma: Scale parameter (normalized_shape)
        beta: Shift parameter (normalized_shape)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
    ):
        """
        Initialize LayerNorm.

        Args:
            normalized_shape: Shape over which to normalize
                If int: normalize over last `normalized_shape` dimensions
                If tuple: exact shape of the normalized dimensions
            eps: Small constant for numerical stability
        """
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        # Learnable parameters
        self.gamma: np.ndarray = np.ones(self.normalized_shape, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(self.normalized_shape, dtype=np.float64)

        # Cache for backward
        self._cache: Optional[Dict[str, np.ndarray]] = None

        # Gradients
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for layer normalization.

        Args:
            x: Input tensor where last dims match normalized_shape

        Returns:
            Normalized tensor of same shape
        """
        x = _ensure_array(x)

        # Dimensions to normalize over
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))

        # Compute mean and variance
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        # Cache for backward
        self._cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "std": std,
            "axes": axes,
            "N": int(np.prod(self.normalized_shape)),
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for layer normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        axes = self._cache["axes"]
        N = self._cache["N"]

        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_norm, axis=axes)
        self.grad_beta = np.sum(grad_output, axis=axes)

        # Gradient w.r.t. x_norm
        dx_norm = grad_output * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3),
            axis=axes,
            keepdims=True,
        )

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * (-1 / std), axis=axes, keepdims=True) + dvar * np.mean(
            -2 * (x - self._cache["mean"]), axis=axes, keepdims=True
        )

        # Gradient w.r.t. x
        grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / N + dmean / N

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        return [self.gamma, self.beta]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        return [self.grad_gamma, self.grad_beta]

    def zero_grad(self) -> None:
        """Reset gradients."""
        self.grad_gamma = None
        self.grad_beta = None


# =============================================================================
# InstanceNorm2d - Normalize per sample per channel
# =============================================================================


class InstanceNorm2d:
    """
    Instance Normalization for 2D inputs (N, C, H, W).

    Normalizes each sample and each channel independently:
        For each (n, c), compute mean/var over (H, W)

    Used primarily in style transfer and image generation.

    Formula:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta (optional, often no affine)

    Attributes:
        num_features: Number of channels (C)
        eps: Small constant for numerical stability
        affine: Whether to use learnable gamma and beta
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Initialize InstanceNorm2d.

        Args:
            num_features: Number of channels (C)
            eps: Small constant for numerical stability
            affine: Whether to use learnable scale and shift
        """
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # Learnable parameters (optional)
        if affine:
            self.gamma: np.ndarray = np.ones(num_features, dtype=np.float64)
            self.beta: np.ndarray = np.zeros(num_features, dtype=np.float64)
        else:
            self.gamma = np.ones(num_features, dtype=np.float64)
            self.beta = np.zeros(num_features, dtype=np.float64)

        # Cache for backward
        self._cache: Optional[Dict[str, np.ndarray]] = None

        # Gradients
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for instance normalization.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        x = _ensure_array(x)
        N, C, H, W = x.shape

        # Compute mean and variance for each sample and channel
        # Shape: (N, C, 1, 1)
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std

        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

        # Cache for backward
        self._cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "std": std,
            "N": H * W,
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for instance normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        N = self._cache["N"]

        if self.affine:
            # Gradient w.r.t. gamma and beta
            self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2, 3))
            self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))

            # Gradient w.r.t. x_norm
            dx_norm = grad_output * self.gamma.reshape(1, -1, 1, 1)
        else:
            dx_norm = grad_output

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3),
            axis=(2, 3),
            keepdims=True,
        )

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * (-1 / std), axis=(2, 3), keepdims=True) + dvar * np.mean(
            -2 * (x - self._cache["mean"]), axis=(2, 3), keepdims=True
        )

        # Gradient w.r.t. x
        grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / N + dmean / N

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        if self.affine:
            return [self.gamma, self.beta]
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        if self.affine:
            return [self.grad_gamma, self.grad_beta]
        return []

    def zero_grad(self) -> None:
        """Reset gradients."""
        self.grad_gamma = None
        self.grad_beta = None


# =============================================================================
# GroupNorm - Normalize across channel groups
# =============================================================================


class GroupNorm:
    """
    Group Normalization for 2D inputs (N, C, H, W).

    Divides channels into groups and normalizes within each group.
    Batch-size independent - works well with small batches.

    Formula:
        For G groups, reshape to (N, G, C//G, H, W)
        Compute mean/var over (C//G, H, W) for each (N, G)
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    Attributes:
        num_groups: Number of groups (G)
        num_channels: Number of channels (C), must be divisible by num_groups
        eps: Small constant for numerical stability
        affine: Whether to use learnable gamma and beta
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Initialize GroupNorm.

        Args:
            num_groups: Number of groups (G)
            num_channels: Number of channels (C)
            eps: Small constant for numerical stability
            affine: Whether to use learnable scale and shift

        Raises:
            ValueError: If num_channels is not divisible by num_groups
        """
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        self.channels_per_group = num_channels // num_groups

        # Learnable parameters
        if affine:
            self.gamma: np.ndarray = np.ones(num_channels, dtype=np.float64)
            self.beta: np.ndarray = np.zeros(num_channels, dtype=np.float64)
        else:
            self.gamma = np.ones(num_channels, dtype=np.float64)
            self.beta = np.zeros(num_channels, dtype=np.float64)

        # Cache for backward
        self._cache: Optional[Dict[str, np.ndarray]] = None

        # Gradients
        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for group normalization.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        x = _ensure_array(x)
        N, C, H, W = x.shape
        G = self.num_groups

        # Reshape to (N, G, C//G, H, W)
        x_grouped = x.reshape(N, G, self.channels_per_group, H, W)

        # Compute mean and variance over (C//G, H, W) for each group
        # Shape: (N, G, 1, 1, 1)
        mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)

        # Normalize
        std = np.sqrt(var + self.eps)
        x_norm_grouped = (x_grouped - mean) / std

        # Reshape back to (N, C, H, W)
        x_norm = x_norm_grouped.reshape(N, C, H, W)

        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

        # Cache for backward
        self._cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "std": std,
            "N": self.channels_per_group * H * W,
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for group normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")

        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        N = self._cache["N"]

        N_batch, C, H, W = x.shape
        G = self.num_groups

        if self.affine:
            # Gradient w.r.t. gamma and beta
            self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2, 3))
            self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))

            # Gradient w.r.t. x_norm
            dx_norm = grad_output * self.gamma.reshape(1, -1, 1, 1)
        else:
            dx_norm = grad_output

        # Reshape for group-wise computation
        x_grouped = x.reshape(N_batch, G, self.channels_per_group, H, W)
        dx_norm_grouped = dx_norm.reshape(N_batch, G, self.channels_per_group, H, W)

        # Gradient w.r.t. variance
        dvar = np.sum(
            dx_norm_grouped * (x_grouped - self._cache["mean"]) * (-0.5) * (std ** -3),
            axis=(2, 3, 4),
            keepdims=True,
        )

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm_grouped * (-1 / std), axis=(2, 3, 4), keepdims=True) + dvar * np.mean(
            -2 * (x_grouped - self._cache["mean"]), axis=(2, 3, 4), keepdims=True
        )

        # Gradient w.r.t. x
        grad_input_grouped = (
            dx_norm_grouped / std
            + dvar * 2 * (x_grouped - self._cache["mean"]) / N
            + dmean / N
        )

        grad_input = grad_input_grouped.reshape(N_batch, C, H, W)

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return learnable parameters."""
        if self.affine:
            return [self.gamma, self.beta]
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return gradients."""
        if self.affine:
            return [self.grad_gamma, self.grad_beta]
        return []

    def zero_grad(self) -> None:
        """Reset gradients."""
        self.grad_gamma = None
        self.grad_beta = None


# =============================================================================
# Registry
# =============================================================================

NORMALIZATION_FUNCTIONS = {
    "batchnorm1d": BatchNorm1d,
    "batchnorm2d": BatchNorm2d,
    "batchnorm": BatchNorm2d,
    "layernorm": LayerNorm,
    "instancenorm2d": InstanceNorm2d,
    "instancenorm": InstanceNorm2d,
    "groupnorm": GroupNorm,
}


def get_normalization(name: str, **kwargs) -> Any:
    """
    Get normalization layer by name.

    Args:
        name: Normalization name (case-insensitive)
        **kwargs: Arguments to pass to normalization constructor

    Returns:
        Normalization layer instance

    Raises:
        ValueError: If normalization name is not recognized
    """
    name_lower = name.lower().replace("-", "").replace("_", "")
    if name_lower not in NORMALIZATION_FUNCTIONS:
        available = ", ".join(NORMALIZATION_FUNCTIONS.keys())
        raise ValueError(f"Unknown normalization: {name}. Available: {available}")
    return NORMALIZATION_FUNCTIONS[name_lower](**kwargs)
