"""
Multi-Layer Perceptron (MLP) implementation with forward and backward propagation.

This module provides a from-scratch implementation of an MLP using NumPy,
with comprehensive gradient computation for backpropagation.

Components:
    LinearLayer: Single linear transformation (y = Wx + b)
    MLP: Multi-layer perceptron combining Linear + Activation layers

Theory:
    Forward: z = Wx + b, a = activation(z)
    Backward: Uses chain rule to compute gradients
        - dL/dW = dL/da * da/dz * dz/dW
        - dL/db = dL/da * da/dz * dz/db
        - dL/dx = dL/da * da/dz * dz/dx

References:
    - Deep Learning (Goodfellow et al.): Chapter 6 - Deep Feedforward Networks
    - Neural Networks and Deep Learning (Nielsen): Chapter 2 - Backpropagation
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np

from .activations import (
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    tanh,
    tanh_grad,
    gelu,
    gelu_grad,
    swish,
    swish_grad,
    get_activation,
)


class LinearLayer:
    """
    Linear (fully connected) layer: y = x @ W + b

    Attributes:
        weight: Weight matrix of shape (in_features, out_features)
        bias: Bias vector of shape (out_features,)
        grad_weight: Gradient of loss w.r.t. weights
        grad_bias: Gradient of loss w.r.t. bias

    Forward:
        output = input @ weight + bias

    Backward:
        grad_input = grad_output @ weight.T
        grad_weight = input.T @ grad_output
        grad_bias = sum(grad_output, axis=0)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize linear layer with Xavier initialization.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        # Xavier/Glorot initialization for better gradient flow
        # std = sqrt(2 / (fan_in + fan_out))
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight: np.ndarray = np.random.randn(in_features, out_features) * std
        self.bias: Optional[np.ndarray] = (
            np.zeros(out_features) if bias else None
        )

        # Gradients (computed in backward)
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Cache for backward pass
        self._input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W + b

        Args:
            x: Input array of shape (batch_size, in_features)

        Returns:
            Output array of shape (batch_size, out_features)
        """
        self._input_cache = x
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, out_features)

        Returns:
            Gradient for previous layer, shape (batch_size, in_features)

        Computes:
            grad_weight = input.T @ grad_output
            grad_bias = sum(grad_output, axis=0)
            grad_input = grad_output @ weight.T
        """
        if self._input_cache is None:
            raise RuntimeError("Must call forward() before backward()")

        # Gradient w.r.t. weights: (in_features, batch_size) @ (batch_size, out_features)
        # = (in_features, out_features)
        self.grad_weight = self._input_cache.T @ grad_output

        # Gradient w.r.t. bias: sum over batch dimension
        if self.bias is not None:
            self.grad_bias = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input: (batch_size, out_features) @ (out_features, in_features)
        # = (batch_size, in_features)
        grad_input = grad_output @ self.weight.T

        return grad_input

    def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get parameters and their gradients.

        Returns:
            List of (parameter, gradient) tuples
        """
        params = [(self.weight, self.grad_weight)]
        if self.bias is not None:
            params.append((self.bias, self.grad_bias))
        return params

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        self.grad_weight = None
        self.grad_bias = None


class ActivationLayer:
    """
    Activation layer wrapper.

    Applies an activation function and its gradient during backprop.

    Forward: a = activation(z)
    Backward: grad_z = grad_a * activation_grad(z)
    """

    def __init__(
        self,
        activation_fn: Callable[[np.ndarray], np.ndarray],
        activation_grad_fn: Callable[[np.ndarray], np.ndarray],
        name: str = "activation",
    ):
        """
        Initialize activation layer.

        Args:
            activation_fn: Forward activation function
            activation_grad_fn: Gradient function (takes pre-activation input)
            name: Name of activation for debugging
        """
        self.activation_fn = activation_fn
        self.activation_grad_fn = activation_grad_fn
        self.name = name
        self._pre_activation: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: apply activation function.

        Args:
            x: Pre-activation input (from linear layer)

        Returns:
            Activated output
        """
        self._pre_activation = x
        return self.activation_fn(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: multiply by activation gradient.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient w.r.t. pre-activation input
        """
        if self._pre_activation is None:
            raise RuntimeError("Must call forward() before backward()")
        # Element-wise multiplication: dL/dz = dL/da * da/dz
        return grad_output * self.activation_grad_fn(self._pre_activation)

    def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Activation layers have no learnable parameters."""
        return []

    def zero_grad(self) -> None:
        """No gradients to reset."""
        pass


class MLP:
    """
    Multi-Layer Perceptron (fully connected neural network).

    Architecture:
        Input -> [Linear -> Activation] x (n_layers-1) -> Linear -> Output

    Example (3-layer MLP with hidden sizes [64, 32]):
        Input(128) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(10) -> Output

    Usage:
        >>> mlp = MLP(input_size=784, hidden_sizes=[64, 32], output_size=10)
        >>> output = mlp.forward(x)  # Forward pass
        >>> grad = mlp.backward(grad_output)  # Backward pass
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        """
        Initialize MLP.

        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output features
            activation: Activation function for hidden layers
            output_activation: Optional activation for output layer
        """
        self.layers: List[Any] = []
        self._activations_cache: List[np.ndarray] = []

        # Get activation functions
        act_fn, act_grad_fn = get_activation(activation)

        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            self.layers.append(
                LinearLayer(layer_sizes[i], layer_sizes[i + 1], bias=True)
            )

            # Add activation (except for last layer unless specified)
            is_last_layer = i == len(layer_sizes) - 2
            if not is_last_layer:
                self.layers.append(
                    ActivationLayer(act_fn, act_grad_fn, name=f"{activation}_{i}")
                )
            elif output_activation is not None:
                out_fn, out_grad_fn = get_activation(output_activation)
                self.layers.append(
                    ActivationLayer(out_fn, out_grad_fn, name=f"{output_activation}_out")
                )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            x: Input of shape (batch_size, input_size)

        Returns:
            Output of shape (batch_size, output_size)
        """
        self._activations_cache = []
        for layer in self.layers:
            x = layer.forward(x)
            self._activations_cache.append(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers (reverse order).

        Uses chain rule: propagate gradients from output to input.

        Args:
            grad_output: Gradient of loss w.r.t. network output

        Returns:
            Gradient w.r.t. network input
        """
        grad = grad_output
        # Iterate in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get all learnable parameters and their gradients.

        Returns:
            Flat list of (parameter, gradient) tuples
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset all gradients."""
        for layer in self.layers:
            layer.zero_grad()

    def get_layer_outputs(self) -> List[np.ndarray]:
        """Get cached layer outputs for debugging/visualization."""
        return self._activations_cache.copy()


class ComputationalGraphVisualizer:
    """
    Visualize computational graph of neural network.

    Creates ASCII representation of the forward/backward pass flow.
    """

    @staticmethod
    def visualize_forward(mlp: MLP) -> str:
        """
        Create ASCII visualization of forward pass.

        Args:
            mlp: MLP instance

        Returns:
            ASCII art string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("FORWARD PASS (Computational Graph)")
        lines.append("=" * 50)

        for i, layer in enumerate(mlp.layers):
            if isinstance(layer, LinearLayer):
                shape = layer.weight.shape
                lines.append(f"  [Linear] in={shape[0]}, out={shape[1]}")
            elif isinstance(layer, ActivationLayer):
                lines.append(f"  [{layer.name.capitalize()}]")
            lines.append("      |")

        lines.append("  [Output]")
        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def visualize_backward(mlp: MLP) -> str:
        """
        Create ASCII visualization of backward pass.

        Args:
            mlp: MLP instance

        Returns:
            ASCII art string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("BACKWARD PASS (Gradient Flow)")
        lines.append("=" * 50)

        lines.append("  [grad_output]")
        for layer in reversed(mlp.layers):
            lines.append("      |")
            if isinstance(layer, LinearLayer):
                lines.append(f"  [Linear] dW, db computed")
            elif isinstance(layer, ActivationLayer):
                lines.append(f"  [{layer.name}] elem-wise mul")

        lines.append("  [grad_input]")
        lines.append("=" * 50)
        return "\n".join(lines)


def numerical_gradient_mlp(
    mlp: MLP,
    x: np.ndarray,
    target: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    param_name: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute numerical gradient for a specific parameter in MLP.

    Used for verifying analytical gradients during backpropagation.

    Args:
        mlp: MLP model
        x: Input data
        target: Target values
        loss_fn: Loss function (pred, target) -> scalar
        param_name: Parameter to check ('weight_i' or 'bias_i')
        eps: Perturbation for finite difference

    Returns:
        Numerical gradient array of same shape as parameter
    """
    # Find the parameter
    layer_idx = int(param_name.split("_")[1])
    param_type = param_name.split("_")[0]

    layer = mlp.layers[layer_idx * 2]  # Linear layers at even indices
    if param_type == "weight":
        param = layer.weight
    else:
        param = layer.bias

    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        # f(x + eps)
        param[idx] = old_val + eps
        pred = mlp.forward(x)
        f_plus = loss_fn(pred, target)

        # f(x - eps)
        param[idx] = old_val - eps
        pred = mlp.forward(x)
        f_minus = loss_fn(pred, target)

        # Restore
        param[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()

    return grad


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Squared Error loss.

    Formula: L = mean((pred - target)^2)

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        Scalar loss value
    """
    return float(np.mean((pred - target) ** 2))


def mse_loss_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss w.r.t. predictions.

    Formula: dL/d(pred) = 2 * (pred - target) / n

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        Gradient array of same shape as pred
    """
    n = pred.size
    return 2.0 * (pred - target) / n
