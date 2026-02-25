"""
Weight initialization methods for neural networks.

This module provides implementations of 5 common initialization strategies:
    - Xavier/Glorot: For Sigmoid/Tanh activations
    - He/Kaiming: For ReLU and variants
    - LSUV: Layer-Sequential Unit-Variance
    - Zero: For comparison (demonstrates symmetry problem)

Theory:
    Proper initialization is critical for:
    1. Avoiding vanishing/exploding gradients
    2. Maintaining activation variance across layers
    3. Ensuring gradient flow in deep networks

    Key insight: Variance of activations should be preserved:
        Var(a^l) = Var(a^{l-1})

    For linear layer y = Wx + b with n_in inputs:
        Var(y) = n_in * Var(W) * Var(x)

    To preserve variance: Var(W) = 1/n_in (He) or 2/(n_in+n_out) (Xavier)

References:
    - Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
    - Delving Deep into Rectifiers (He et al., 2015)
    - All You Need is a Good Init (Mishkin & Matas, 2015)
"""

from typing import Literal, Tuple, Optional, Callable
import numpy as np

# Type aliases
ArrayLike = np.ndarray
InitMode = Literal["xavier_uniform", "xavier_normal", "he_uniform", "he_normal", "kaiming_uniform", "kaiming_normal", "lsuv", "zero"]


def xavier_uniform(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.

    Formula: W ~ U(-a, a) where a = sqrt(6 / (fan_in + fan_out))

    Designed for sigmoid/tanh activations. Preserves variance of activations
    and gradients when using symmetric activation functions.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)

    Example:
        >>> W = xavier_uniform(784, 256)
        >>> W.shape
        (784, 256)

    Notes:
        - Best for sigmoid, tanh activations
        - For ReLU networks, prefer He initialization
        - Variance: 2 / (fan_in + fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std  # Uniform bound for same variance
    return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float64)


def xavier_normal(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.

    Formula: W ~ N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))

    Normal distribution variant of Xavier initialization.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)

    Notes:
        - Normal variant may have slightly different properties than uniform
        - Both maintain variance ~ 2/(fan_in + fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, size=(fan_in, fan_out)).astype(np.float64)


def he_uniform(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    He uniform initialization (Kaiming with fan_in mode).

    Formula: W ~ U(-a, a) where a = sqrt(6 / fan_in)

    Designed for ReLU and variants. Accounts for ReLU zeroing half the
    activations by using fan_in only (not fan_in + fan_out).

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)

    Notes:
        - Best for ReLU, Leaky ReLU, GELU activations
        - Variance: 2 / fan_in (accounts for ReLU killing half the units)
        - Also known as Kaiming uniform with mode='fan_in'
    """
    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / fan_in)
    bound = np.sqrt(3.0) * std
    return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float64)


def he_normal(fan_in: int, fan_out: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    He normal initialization (Kaiming with fan_in mode).

    Formula: W ~ N(0, std^2) where std = sqrt(2 / fan_in)

    Normal distribution variant of He initialization.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)

    Notes:
        - Recommended for deep networks with ReLU
        - Default choice in PyTorch for nn.Linear with ReLU
    """
    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=(fan_in, fan_out)).astype(np.float64)


def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: Literal["relu", "leaky_relu"] = "leaky_relu",
    a: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Kaiming uniform initialization (generalized He).

    Formula: W ~ U(-bound, bound) where bound = sqrt(6 / (gain^2 * fan))

    Generalizes He initialization with configurable mode and nonlinearity.

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        mode: 'fan_in' preserves variance in forward pass, 'fan_out' for backward
        nonlinearity: Activation type ('relu' or 'leaky_relu')
        a: Negative slope for leaky_relu (default: 0 for standard ReLU)
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)

    Notes:
        - mode='fan_in': Good for forward pass variance
        - mode='fan_out': Good for backward pass gradient variance
        - For LeakyReLU with slope 0.01, use a=0.01
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute gain based on nonlinearity
    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    # Select fan mode
    fan = fan_in if mode == "fan_in" else fan_out

    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float64)


def kaiming_normal(
    fan_in: int,
    fan_out: int,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: Literal["relu", "leaky_relu"] = "leaky_relu",
    a: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Kaiming normal initialization (generalized He).

    Formula: W ~ N(0, std^2) where std = gain / sqrt(fan)

    Args:
        fan_in: Number of input units
        fan_out: Number of output units
        mode: 'fan_in' or 'fan_out'
        nonlinearity: Activation type ('relu' or 'leaky_relu')
        a: Negative slope for leaky_relu
        rng: Optional numpy random generator for reproducibility

    Returns:
        Weight matrix of shape (fan_in, fan_out)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute gain based on nonlinearity
    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    fan = fan_in if mode == "fan_in" else fan_out
    std = gain / np.sqrt(fan)
    return rng.normal(0, std, size=(fan_in, fan_out)).astype(np.float64)


def zero_init(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Zero initialization (for demonstration only).

    Returns:
        Zero matrix of shape (fan_in, fan_out)

    Notes:
        - DO NOT USE for weights - causes symmetry problem
        - All neurons learn identical features
        - Gradients will be identical, no learning diversity
        - Included for educational comparison only
        - Safe to use for bias initialization
    """
    return np.zeros((fan_in, fan_out), dtype=np.float64)


def lsuv_init(
    weight: np.ndarray,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    target_variance: float = 1.0,
    max_iterations: int = 10,
    tol: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, int]:
    """
    Layer-Sequential Unit-Variance (LSUV) initialization.

    Iteratively scales weights to achieve unit variance of activations.
    Works with any architecture, including very deep or unusual networks.

    Algorithm:
        1. Initialize weights with orthogonal or Gaussian
        2. Forward pass with dummy input
        3. Compute variance of output
        4. Scale weights: W = W * sqrt(target_var / current_var)
        5. Repeat until variance is within tolerance

    Args:
        weight: Initial weight matrix of shape (fan_in, fan_out)
        forward_fn: Function that takes input and returns layer output
                    Signature: forward_fn(x) -> output
        target_variance: Target variance for output activations (default: 1.0)
        max_iterations: Maximum iterations before stopping (default: 10)
        tol: Tolerance for variance convergence (default: 0.1)
        rng: Optional numpy random generator

    Returns:
        Tuple of (initialized weight matrix, number of iterations used)

    Example:
        >>> W = np.random.randn(784, 256).astype(np.float64)
        >>> def forward_fn(x):
        ...     return x @ W
        >>> W_init, iters = lsuv_init(W, forward_fn)
        >>> print(f"Converged in {iters} iterations")

    Notes:
        - Usually converges in 1-5 iterations
        - Works well for CNNs and very deep networks
        - More expensive than analytical methods (requires forward pass)
        - Paper: "All You Need is a Good Init" (Mishkin & Matas, 2015)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Work directly on the input weight array (in-place modification)
    # This is important because forward_fn may capture weight by reference

    # Create dummy input with unit variance
    fan_in = weight.shape[0]
    batch_size = 1000
    dummy_input = rng.standard_normal((batch_size, fan_in)).astype(np.float64)

    for iteration in range(max_iterations):
        # Forward pass
        output = forward_fn(dummy_input)

        # Compute variance of output (exclude batch dimension)
        current_var = np.var(output)

        # Check convergence
        if abs(current_var - target_variance) < tol:
            return weight, iteration + 1

        # Avoid division by zero
        if current_var < 1e-10:
            # If variance is too small, use a default scale
            scale = 10.0  # Large scale to boost variance
        else:
            scale = np.sqrt(target_variance / current_var)

        # Scale weights in-place
        weight *= scale

    # Return even if not converged within max_iterations
    return weight, max_iterations


def compute_fan(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for a weight tensor.

    For 2D weight matrix (linear layer): (in_features, out_features)
    For 4D weight tensor (conv layer): (out_channels, in_channels, kH, kW)

    Args:
        shape: Shape of the weight tensor

    Returns:
        Tuple of (fan_in, fan_out)
    """
    if len(shape) == 2:
        # Linear layer: (in_features, out_features)
        fan_in, fan_out = shape
    elif len(shape) >= 3:
        # Convolutional layer: (out_channels, in_channels, ...)
        receptive_field = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    else:
        raise ValueError(f"Invalid shape for weight tensor: {shape}")

    return int(fan_in), int(fan_out)


def get_initializer(name: str) -> Callable:
    """
    Get initialization function by name.

    Args:
        name: Initialization method name

    Returns:
        Initialization function

    Supported names:
        - 'xavier_uniform', 'glorot_uniform'
        - 'xavier_normal', 'glorot_normal'
        - 'he_uniform'
        - 'he_normal'
        - 'kaiming_uniform'
        - 'kaiming_normal'
        - 'zero'

    Example:
        >>> init_fn = get_initializer('he_normal')
        >>> W = init_fn(784, 256)
    """
    init_map = {
        "xavier_uniform": xavier_uniform,
        "glorot_uniform": xavier_uniform,
        "xavier_normal": xavier_normal,
        "glorot_normal": xavier_normal,
        "he_uniform": he_uniform,
        "he_normal": he_normal,
        "kaiming_uniform": kaiming_uniform,
        "kaiming_normal": kaiming_normal,
        "zero": zero_init,
    }

    if name not in init_map:
        available = ", ".join(sorted(init_map.keys()))
        raise ValueError(f"Unknown initializer: {name}. Available: {available}")

    return init_map[name]


def init_bias(size: int, method: Literal["zeros", "ones", "small"] = "zeros") -> np.ndarray:
    """
    Initialize bias vector.

    Args:
        size: Size of bias vector
        method: Initialization method:
            - 'zeros': All zeros (common default)
            - 'ones': All ones
            - 'small': Small positive values (0.01)

    Returns:
        Bias vector of shape (size,)

    Notes:
        - Zeros is most common and generally safe
        - Small positive can help with ReLU dying units initially
        - Unlike weights, bias initialization is less critical
    """
    if method == "zeros":
        return np.zeros(size, dtype=np.float64)
    elif method == "ones":
        return np.ones(size, dtype=np.float64)
    elif method == "small":
        return np.full(size, 0.01, dtype=np.float64)
    else:
        raise ValueError(f"Unknown bias init method: {method}")


# Registry for easy access
INITIALIZERS = {
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "kaiming_uniform": kaiming_uniform,
    "kaiming_normal": kaiming_normal,
    "zero": zero_init,
}
