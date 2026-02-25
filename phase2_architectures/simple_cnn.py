"""
Simple CNN and ResNet-like architectures for CIFAR10 classification.

This module provides:
    - SimpleCNN: Basic CNN with Conv-BN-ReLU-Pool pattern
    - ResidualBlock: Basic residual block with skip connections
    - ResNetSmall: Small ResNet for CIFAR10 (similar to ResNet18)

Theory:
    ResNet Skip Connections:
        output = F(x) + x  (identity mapping)
        This allows gradients to flow directly through the network,
        enabling training of very deep networks (100+ layers).

    Batch Normalization:
        normalizes activations across the batch dimension,
        reducing internal covariate shift and allowing higher learning rates.

Architecture for CIFAR10:
    Input: 32x32x3 images
    SimpleCNN: Conv -> Pool -> Conv -> Pool -> FC -> FC (10 classes)
    ResNetSmall: Conv -> ResBlock x 4 -> AvgPool -> FC (10 classes)

References:
    - Deep Residual Learning for Image Recognition (He et al., 2015)
    - Batch Normalization (Ioffe & Szegedy, 2015)
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# Import from our implementations
from .cnn_layers import (
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    Flatten,
    compute_output_shape,
)

from phase1_basics.activations import relu, relu_grad
from phase1_basics.weight_init import he_normal


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
        self.gamma: np.ndarray = np.ones(num_features)
        self.beta: np.ndarray = np.zeros(num_features)

        # Running statistics (for inference)
        self.running_mean: np.ndarray = np.zeros(num_features)
        self.running_var: np.ndarray = np.ones(num_features)

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
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)

        # Cache for backward
        self._cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "var": var,
            "std": np.sqrt(var + self.eps),
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

        batch, channels, height, width = x.shape
        n = batch * height * width

        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_norm, axis=(0, 2, 3))
        self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))

        # Gradient w.r.t. x_norm
        dx_norm = grad_output * self.gamma.reshape(1, -1, 1, 1)

        # Gradient w.r.t. variance
        dvar = np.sum(dx_norm * (x - self._cache["mean"]) * (-0.5) * (std ** -3), axis=(0, 2, 3), keepdims=True)

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * (-1 / std), axis=(0, 2, 3), keepdims=True) + \
                dvar * np.mean(-2 * (x - self._cache["mean"]), axis=(0, 2, 3), keepdims=True)

        # Gradient w.r.t. x
        grad_input = dx_norm / std + dvar * 2 * (x - self._cache["mean"]) / n + dmean / n

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


class ReLULayer:
    """ReLU activation layer wrapper."""

    def __init__(self):
        self._cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return relu(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Must call forward before backward")
        return grad_output * relu_grad(self._cache)

    def parameters(self) -> List[np.ndarray]:
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        return []

    def zero_grad(self) -> None:
        pass


class SimpleCNN:
    """
    Simple CNN for CIFAR10 classification.

    Architecture:
        Conv(3, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Conv(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Conv(64, 128, 3, pad=1) -> BN -> ReLU -> MaxPool(2)
        Flatten -> FC(128*4*4, 256) -> ReLU -> FC(256, 10)

    Input: (batch, 3, 32, 32)
    Output: (batch, 10)

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> x = np.random.randn(16, 3, 32, 32)
        >>> out = model.forward(x)
        >>> out.shape
        (16, 10)
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize SimpleCNN.

        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes

        # Layer 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Conv -> BN -> ReLU -> Pool
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: Conv -> BN -> ReLU -> Pool
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = ReLULayer()
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        # Flatten
        self.flatten = Flatten()

        # Fully connected layers (imported from phase1_basics)
        from phase1_basics.mlp import LinearLayer
        self.fc1 = LinearLayer(128 * 4 * 4, 256)
        self.relu_fc = ReLULayer()
        self.fc2 = LinearLayer(256, num_classes)

        # Store all layers
        self._layers = [
            self.conv1, self.bn1, self.relu1, self.pool1,
            self.conv2, self.bn2, self.relu2, self.pool2,
            self.conv3, self.bn3, self.relu3, self.pool3,
            self.flatten, self.fc1, self.relu_fc, self.fc2,
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input images of shape (batch, 3, 32, 32)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Layer 1
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        # Layer 2
        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        # Layer 3
        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)

        # Classifier
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu_fc.forward(x)
        x = self.fc2.forward(x)

        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the network.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        grad = grad_output

        # Reverse order for backward
        grad = self.fc2.backward(grad)
        grad = self.relu_fc.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)

        grad = self.pool3.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.bn3.backward(grad)
        grad = self.conv3.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.bn2.backward(grad)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)

        return grad

    def parameters(self) -> List[np.ndarray]:
        """Return all learnable parameters."""
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return all gradients."""
        grads = []
        for layer in self._layers:
            # Handle layers with gradients() method
            if hasattr(layer, 'gradients') and callable(getattr(layer, 'gradients')):
                grads.extend(layer.gradients())
            # Handle LinearLayer from phase1_basics (no gradients method)
            elif hasattr(layer, 'grad_weight'):
                grads.append((layer.grad_weight, layer.grad_bias))
            else:
                grads.append(None)
        return grads

    def zero_grad(self) -> None:
        """Reset all gradients."""
        for layer in self._layers:
            layer.zero_grad()

    def train(self) -> None:
        """Set all layers to training mode."""
        for layer in self._layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode."""
        for layer in self._layers:
            if hasattr(layer, 'eval'):
                layer.eval()


class ResidualBlock:
    """
    Basic residual block for ResNet.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> + x -> ReLU -> output
        |___________________________________________|

    The skip connection allows gradients to flow directly,
    enabling training of very deep networks.

    Example:
        >>> block = ResidualBlock(64, 64, stride=1)
        >>> x = np.random.randn(16, 64, 32, 32)
        >>> out = block.forward(x)
        >>> out.shape
        (16, 64, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        """
        Initialize ResidualBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Main path
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLULayer()

        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu2 = ReLULayer()

        # Shortcut connection
        self.shortcut: Optional[Conv2d] = None
        self.shortcut_bn: Optional[BatchNorm2d] = None

        if stride != 1 or in_channels != out_channels:
            # Need projection shortcut
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2d(out_channels)

        self._cache_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height/stride, width/stride)
        """
        self._cache_input = x

        # Main path
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # Shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut.forward(x)
            shortcut = self.shortcut_bn.forward(shortcut)
        else:
            shortcut = x

        # Add skip connection
        out = out + shortcut

        # Final activation
        out = self.relu2.forward(out)

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass with gradient routing through skip connection.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        # Gradient through final ReLU
        grad = self.relu2.backward(grad_output)

        # Gradient through main path
        grad_main = self.bn2.backward(grad)
        grad_main = self.conv2.backward(grad_main)

        grad_main = self.relu1.backward(grad_main)
        grad_main = self.bn1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)

        # Gradient through shortcut
        if self.shortcut is not None:
            grad_shortcut = self.shortcut_bn.backward(grad)
            grad_shortcut = self.shortcut.backward(grad_shortcut)
        else:
            grad_shortcut = grad

        # Add gradients (from skip connection addition)
        grad_input = grad_main + grad_shortcut

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        if self.shortcut is not None:
            params.extend(self.shortcut.parameters())
            params.extend(self.shortcut_bn.parameters())
        return params

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return all gradients."""
        grads = []
        grads.extend(self.conv1.gradients())
        grads.extend(self.bn1.gradients())
        grads.extend(self.conv2.gradients())
        grads.extend(self.bn2.gradients())
        if self.shortcut is not None:
            grads.extend(self.shortcut.gradients())
            grads.extend(self.shortcut_bn.gradients())
        return grads

    def zero_grad(self) -> None:
        """Reset all gradients."""
        self.conv1.zero_grad()
        self.bn1.zero_grad()
        self.conv2.zero_grad()
        self.bn2.zero_grad()
        if self.shortcut is not None:
            self.shortcut.zero_grad()
            self.shortcut_bn.zero_grad()


class ResNetSmall:
    """
    Small ResNet for CIFAR10 classification (similar to ResNet18).

    Architecture:
        Conv(3, 64, 3, pad=1) -> BN -> ReLU
        ResBlock(64, 64) x 2
        ResBlock(64, 128, stride=2) -> ResBlock(128, 128)
        ResBlock(128, 256, stride=2) -> ResBlock(256, 256)
        ResBlock(256, 512, stride=2) -> ResBlock(512, 512)
        AvgPool(4) -> FC(512, 10)

    Input: (batch, 3, 32, 32)
    Output: (batch, num_classes)

    Expected accuracy on CIFAR10: >85% with proper training

    Example:
        >>> model = ResNetSmall(num_classes=10)
        >>> x = np.random.randn(16, 3, 32, 32)
        >>> out = model.forward(x)
        >>> out.shape
        (16, 10)
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize ResNetSmall.

        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = ReLULayer()

        # Residual blocks
        # Layer 1: 64 channels
        self.block1_1 = ResidualBlock(64, 64, stride=1)
        self.block1_2 = ResidualBlock(64, 64, stride=1)

        # Layer 2: 128 channels
        self.block2_1 = ResidualBlock(64, 128, stride=2)
        self.block2_2 = ResidualBlock(128, 128, stride=1)

        # Layer 3: 256 channels
        self.block3_1 = ResidualBlock(128, 256, stride=2)
        self.block3_2 = ResidualBlock(256, 256, stride=1)

        # Layer 4: 512 channels
        self.block4_1 = ResidualBlock(256, 512, stride=2)
        self.block4_2 = ResidualBlock(512, 512, stride=1)

        # Global average pooling and classifier
        self.avgpool = AvgPool2d(kernel_size=4, stride=4)
        self.flatten = Flatten()

        from phase1_basics.mlp import LinearLayer
        self.fc = LinearLayer(512, num_classes)

        # Store all layers for easy access
        self._layers = [
            self.conv1, self.bn1, self.relu1,
            self.block1_1, self.block1_2,
            self.block2_1, self.block2_2,
            self.block3_1, self.block3_2,
            self.block4_1, self.block4_2,
            self.avgpool, self.flatten, self.fc,
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet.

        Args:
            x: Input images of shape (batch, 3, 32, 32)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Initial convolution
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)

        # Residual blocks
        x = self.block1_1.forward(x)
        x = self.block1_2.forward(x)

        x = self.block2_1.forward(x)
        x = self.block2_2.forward(x)

        x = self.block3_1.forward(x)
        x = self.block3_2.forward(x)

        x = self.block4_1.forward(x)
        x = self.block4_2.forward(x)

        # Global average pooling
        x = self.avgpool.forward(x)
        x = self.flatten.forward(x)

        # Classifier
        x = self.fc.forward(x)

        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through ResNet.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        grad = grad_output

        grad = self.fc.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.avgpool.backward(grad)

        grad = self.block4_2.backward(grad)
        grad = self.block4_1.backward(grad)

        grad = self.block3_2.backward(grad)
        grad = self.block3_1.backward(grad)

        grad = self.block2_2.backward(grad)
        grad = self.block2_1.backward(grad)

        grad = self.block1_2.backward(grad)
        grad = self.block1_1.backward(grad)

        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.conv1.backward(grad)

        return grad

    def parameters(self) -> List[np.ndarray]:
        """Return all learnable parameters."""
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return all gradients."""
        grads = []
        for layer in self._layers:
            # Handle layers with gradients() method
            if hasattr(layer, 'gradients') and callable(getattr(layer, 'gradients')):
                grads.extend(layer.gradients())
            # Handle LinearLayer from phase1_basics (no gradients method)
            elif hasattr(layer, 'grad_weight'):
                grads.append((layer.grad_weight, layer.grad_bias))
            else:
                grads.append(None)
        return grads

    def zero_grad(self) -> None:
        """Reset all gradients."""
        for layer in self._layers:
            layer.zero_grad()

    def train(self) -> None:
        """Set all layers to training mode."""
        for layer in self._layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode."""
        for layer in self._layers:
            if hasattr(layer, 'eval'):
                layer.eval()


def count_parameters(model) -> int:
    """
    Count total number of learnable parameters in a model.

    Handles both:
    - New style: parameters() returns list of np.ndarray
    - Old style: parameters() returns list of (param, grad) tuples

    Args:
        model: Model with parameters() method

    Returns:
        Total number of parameters
    """
    total = 0
    for p in model.parameters():
        if isinstance(p, np.ndarray):
            total += p.size
        elif isinstance(p, tuple):
            # Old style: (param, grad) tuple
            param = p[0]
            if isinstance(param, np.ndarray):
                total += param.size
        elif hasattr(p, 'size'):
            total += p.size
    return total


def get_model_info(model) -> Dict[str, Any]:
    """
    Get model information.

    Args:
        model: Model instance

    Returns:
        Dictionary with model information
    """
    params = model.parameters()
    total_params = sum(p.size for p in params)

    return {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "num_layers": len(model._layers) if hasattr(model, '_layers') else 'N/A',
    }
