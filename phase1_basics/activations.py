"""
Neural network activation functions.

This module provides implementations of 6 common activation functions
with both forward and backward (gradient) computations using NumPy.

Functions:
    sigmoid: Sigmoid activation (σ(x) = 1/(1+e^(-x)))
    tanh: Hyperbolic tangent
    relu: Rectified Linear Unit
    leaky_relu: Leaky ReLU with configurable slope
    gelu: Gaussian Error Linear Unit
    swish: Swish/SiLU activation (x * sigmoid(x))

Each function has a corresponding _grad function for computing derivatives.

References:
    - GELU: https://arxiv.org/abs/1606.08415
    - Swish: https://arxiv.org/abs/1710.05941
"""

import math
from typing import Union

import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, float, int]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array if not already."""
    return np.asarray(x, dtype=np.float64)


# =============================================================================
# Sigmoid
# =============================================================================


def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Sigmoid activation function.

    Formula: σ(x) = 1 / (1 + exp(-x))

    Derivative: σ'(x) = σ(x) * (1 - σ(x))

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape as input with sigmoid applied

    Notes:
        - Output range: (0, 1)
        - Prone to vanishing gradient for large |x|
        - At x=0, output is 0.5 and gradient is 0.25
        - Use GELU or Swish for deep networks to avoid vanishing gradient

    Example:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.731..., 0.268...])
    """
    x = _ensure_array(x)
    # Clip to prevent overflow in exp
    clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_grad(x: ArrayLike) -> np.ndarray:
    """
    Derivative of sigmoid activation.

    Formula: σ'(x) = σ(x) * (1 - σ(x))

    At x=0: σ'(0) = 0.5 * (1 - 0.5) = 0.25

    Args:
        x: Input array (pre-activation values)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> sigmoid_grad(0)
        0.25
    """
    s = sigmoid(x)
    return s * (1.0 - s)


# =============================================================================
# Tanh (Hyperbolic Tangent)
# =============================================================================


def tanh(x: ArrayLike) -> np.ndarray:
    """
    Hyperbolic tangent activation function.

    Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Derivative: tanh'(x) = 1 - tanh(x)²

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape as input with tanh applied

    Notes:
        - Output range: (-1, 1)
        - Zero-centered (unlike sigmoid)
        - Still prone to vanishing gradient for large |x|

    Example:
        >>> tanh(np.array([0, 1, -1]))
        array([0., 0.761..., -0.761...])
    """
    x = _ensure_array(x)
    return np.tanh(x)


def tanh_grad(x: ArrayLike) -> np.ndarray:
    """
    Derivative of tanh activation.

    Formula: tanh'(x) = 1 - tanh(x)²

    At x=0: tanh'(0) = 1 - 0² = 1

    Args:
        x: Input array (pre-activation values)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> tanh_grad(0)
        1.0
    """
    t = tanh(x)
    return 1.0 - t**2


# =============================================================================
# ReLU (Rectified Linear Unit)
# =============================================================================


def relu(x: ArrayLike) -> np.ndarray:
    """
    Rectified Linear Unit activation function.

    Formula: ReLU(x) = max(0, x)

    Derivative: ReLU'(x) = 1 if x > 0, else 0

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape as input with ReLU applied

    Notes:
        - Output range: [0, +∞)
        - Most commonly used in modern neural networks
        - Computationally efficient
        - Suffers from "dying ReLU" problem (neurons can become inactive)
        - Gradient is 0 for all x < 0, 1 for all x > 0

    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0., 0., 0., 1., 2.])
    """
    x = _ensure_array(x)
    return np.maximum(0, x)


def relu_grad(x: ArrayLike) -> np.ndarray:
    """
    Derivative of ReLU activation.

    Formula: ReLU'(x) = 1 if x > 0, else 0

    Note: At x=0, the gradient is technically undefined (subgradient).
    We use 0 as the convention (common in deep learning frameworks).

    Args:
        x: Input array (pre-activation values)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> relu_grad(np.array([-1, 0, 1]))
        array([0., 0., 1.])
    """
    x = _ensure_array(x)
    return (x > 0).astype(np.float64)


# =============================================================================
# Leaky ReLU
# =============================================================================


def leaky_relu(x: ArrayLike, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Formula: LeakyReLU(x) = x if x > 0, else α*x

    Derivative: LeakyReLU'(x) = 1 if x > 0, else α

    Args:
        x: Input array of any shape
        alpha: Slope for negative values (default: 0.01)

    Returns:
        Array of same shape as input with Leaky ReLU applied

    Notes:
        - Output range: (-∞, +∞)
        - Addresses "dying ReLU" problem by having small gradient for x < 0
        - α is typically set to 0.01

    Example:
        >>> leaky_relu(np.array([-2, -1, 0, 1, 2]), alpha=0.1)
        array([-0.2, -0.1, 0., 1., 2.])
    """
    x = _ensure_array(x)
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x: ArrayLike, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU activation.

    Formula: LeakyReLU'(x) = 1 if x > 0, else α

    Args:
        x: Input array (pre-activation values)
        alpha: Slope for negative values (default: 0.01)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> leaky_relu_grad(np.array([-1, 0, 1]), alpha=0.1)
        array([0.1, 0.1, 1.])
    """
    x = _ensure_array(x)
    return np.where(x > 0, 1.0, alpha)


# =============================================================================
# GELU (Gaussian Error Linear Unit)
# =============================================================================


def gelu(x: ArrayLike, approximate: bool = True) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation function.

    Exact formula: GELU(x) = x * Φ(x)
    where Φ(x) is the CDF of standard normal distribution.

    Approximation (default): GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Args:
        x: Input array of any shape
        approximate: If True, use tanh approximation (default: True)

    Returns:
        Array of same shape as input with GELU applied

    Notes:
        - Used in transformer models (BERT, GPT, etc.)
        - Smoother than ReLU, non-monotonic
        - At x=0: GELU(0) = 0 (Φ(0) = 0.5, so 0 * 0.5 = 0)
        - Approximation error is typically < 0.01%

    References:
        - Gaussian Error Linear Units (GELU): https://arxiv.org/abs/1606.08415

    Example:
        >>> gelu(0)
        0.0
        >>> gelu(1)
        0.841...
    """
    x = _ensure_array(x)

    if approximate:
        # Tanh approximation: faster, accurate to ~0.01%
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
    else:
        # Exact using error function: erf(x) = 2/√π * ∫₀ˣ e^(-t²) dt
        # Φ(x) = 0.5 * (1 + erf(x/√2))
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_grad(x: ArrayLike, approximate: bool = True) -> np.ndarray:
    """
    Derivative of GELU activation.

    Exact derivative:
        GELU'(x) = Φ(x) + x * φ(x)
    where φ(x) is the PDF of standard normal: φ(x) = exp(-x²/2) / √(2π)

    Approximation derivative:
        GELU'(x) ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * (√(2/π) * (1 + 0.134145 * x²))

    Args:
        x: Input array (pre-activation values)
        approximate: If True, use tanh approximation derivative (default: True)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> gelu_grad(0)
        0.5
    """
    x = _ensure_array(x)

    if approximate:
        # Derivative of tanh approximation
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        # sech²(x) = 1 - tanh²(x)
        sech_squared = 1.0 - tanh_inner**2
        # d/dx of (x + 0.044715 * x³) = 1 + 0.134145 * x²
        d_inner = 1.0 + 0.134145 * x**2
        # Product rule: d/dx[0.5 * x * (1 + tanh)]
        return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * sqrt_2_over_pi * d_inner
    else:
        # Exact derivative using PDF
        phi_x = np.exp(-(x**2) / 2.0) / math.sqrt(2.0 * math.pi)
        cdf_x = 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
        return cdf_x + x * phi_x


def _erf(x: np.ndarray) -> np.ndarray:
    """
    Error function approximation.

    erf(x) = 2/√π * ∫₀ˣ e^(-t²) dt

    Uses Abramowitz and Stegun approximation (equation 7.1.26).
    Maximum error: 1.5×10⁻⁷

    Args:
        x: Input array

    Returns:
        Error function values
    """
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # Abramowitz and Stegun approximation
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x**2))

    return sign * y


# =============================================================================
# Swish (SiLU - Sigmoid Linear Unit)
# =============================================================================


def swish(x: ArrayLike) -> np.ndarray:
    """
    Swish activation function (also known as SiLU).

    Formula: Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Derivative: Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                          = Swish(x) / x + sigmoid(x) * (1 - Swish(x))
                          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape as input with Swish applied

    Notes:
        - Self-gated activation (smooth, non-monotonic)
        - Outperforms ReLU in deep networks
        - Output range: approximately (-0.28, +∞)
        - Minimum at x ≈ -1.278, Swish(x) ≈ -0.278

    References:
        - Searching for Activation Functions: https://arxiv.org/abs/1710.05941

    Example:
        >>> swish(0)
        0.0
        >>> swish(1)
        0.731...
    """
    x = _ensure_array(x)
    return x * sigmoid(x)


def swish_grad(x: ArrayLike) -> np.ndarray:
    """
    Derivative of Swish activation.

    Formula: Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                       = sigmoid(x) * (1 + x - x * sigmoid(x))

    At x=0: Swish'(0) = 0.5 (since sigmoid(0) = 0.5)

    Args:
        x: Input array (pre-activation values)

    Returns:
        Gradient array of same shape as input

    Example:
        >>> swish_grad(0)
        0.5
    """
    x = _ensure_array(x)
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


# =============================================================================
# Utility function for gradient checking
# =============================================================================


def numerical_gradient(
    func, x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.

    Formula: ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / (2ε)

    Args:
        func: Function to differentiate (takes array, returns scalar)
        x: Input array
        eps: Small perturbation for finite difference (default: 1e-5)

    Returns:
        Numerical gradient array of same shape as x

    Example:
        >>> numerical_gradient(lambda x: np.sum(sigmoid(x)), np.array([1.0]))
        array([0.196...])
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        f_plus = func(x)

        x[idx] = old_val - eps
        f_minus = func(x)

        x[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()

    return grad


# =============================================================================
# Activation function registry for easy access
# =============================================================================

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "gelu": (gelu, gelu_grad),
    "swish": (swish, swish_grad),
}


def get_activation(name: str):
    """
    Get activation function and its gradient by name.

    Args:
        name: Name of activation function (sigmoid, tanh, relu, leaky_relu, gelu, swish)

    Returns:
        Tuple of (forward_fn, backward_fn)

    Raises:
        ValueError: If activation name is not recognized

    Example:
        >>> forward, backward = get_activation("relu")
        >>> forward(np.array([-1, 0, 1]))
        array([0., 0., 1.])
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
