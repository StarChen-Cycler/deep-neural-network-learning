"""
Neural network optimizers for gradient descent.

This module provides implementations of 6 common optimizers:
    - SGD: Stochastic Gradient Descent
    - Momentum: SGD with momentum
    - Nesterov: Nesterov accelerated gradient
    - AdaGrad: Adaptive gradient algorithm
    - RMSprop: Root mean square propagation
    - Adam: Adaptive moment estimation

Each optimizer follows the pattern:
    - step(params, grads) -> updates parameters in-place

Theory:
    Gradient descent minimizes loss L(θ) by iteratively updating:
        θ = θ - lr * ∇L(θ)

    Variants add momentum, adaptive learning rates, or both.

References:
    - An overview of gradient descent optimization algorithms: https://arxiv.org/abs/1609.04747
    - Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Type alias for parameters
Params = List[Tuple[np.ndarray, Optional[np.ndarray]]]


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Formula: θ = θ - lr * ∇L(θ)

    This is the simplest form of gradient descent. Updates parameters
    in the opposite direction of the gradient.

    Args:
        lr: Learning rate (default: 0.01)

    Example:
        >>> optimizer = SGD(lr=0.01)
        >>> params = mlp.parameters()  # List of (param, grad) tuples
        >>> optimizer.step(params)

    Notes:
        - Simple but can be slow to converge
        - Sensitive to learning rate choice
        - Can oscillate in ravines
    """

    def __init__(self, lr: float = 0.01):
        """Initialize SGD optimizer."""
        self.lr = lr

    def step(self, params: Params) -> None:
        """
        Update parameters using gradient descent.

        Args:
            params: List of (parameter, gradient) tuples
        """
        for param, grad in params:
            if grad is not None:
                param -= self.lr * grad

    def zero_grad(self) -> None:
        """SGD has no internal state to reset."""
        pass


class Momentum:
    """
    SGD with momentum optimizer.

    Formula:
        v = μ * v - lr * ∇L(θ)
        θ = θ + v

    Momentum accumulates a velocity vector in directions of persistent
    reduction, dampening oscillations.

    Args:
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient (default: 0.9)

    Example:
        >>> optimizer = Momentum(lr=0.01, momentum=0.9)

    Notes:
        - Accelerates convergence in consistent gradient directions
        - Dampens oscillations in ravines
        - μ=0.9 is a common choice (10x speedup in smooth directions)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        """Initialize Momentum optimizer."""
        self.lr = lr
        self.momentum = momentum
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self, params: Params) -> None:
        """
        Update parameters using momentum.

        Args:
            params: List of (parameter, gradient) tuples
        """
        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize velocity if needed
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param)

                # Update velocity: v = momentum * v - lr * grad
                self.velocities[i] = (
                    self.momentum * self.velocities[i] - self.lr * grad
                )

                # Update parameter: θ = θ + v
                param += self.velocities[i]

    def zero_grad(self) -> None:
        """Reset velocities (usually not needed between batches)."""
        pass


class Nesterov:
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    Formula:
        v_prev = v
        v = μ * v - lr * ∇L(θ + μ * v)
        θ = θ + v

    Nesterov momentum computes gradient at the "lookahead" position,
    providing a corrective term to momentum.

    Args:
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient (default: 0.9)

    Example:
        >>> optimizer = Nesterov(lr=0.01, momentum=0.9)

    Notes:
        - More responsive than standard momentum
        - Better convergence for convex functions (O(1/t²) vs O(1/t))
        - In practice: computes gradient at (θ + μ * v)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        """Initialize Nesterov optimizer."""
        self.lr = lr
        self.momentum = momentum
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self, params: Params) -> None:
        """
        Update parameters using Nesterov accelerated gradient.

        Note: For simplicity, this implementation uses the standard
        momentum update but with a different velocity formula.
        True NAG requires computing gradient at lookahead position.

        The equivalent formulation:
            v = μ * v - lr * grad
            θ = θ + μ * v - lr * grad
        can be simplified to:
            v = μ * v - lr * grad
            θ = θ + v

        Args:
            params: List of (parameter, gradient) tuples
        """
        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize velocity if needed
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param)

                # Nesterov update:
                # The gradient is already computed at current position
                # We apply momentum to the velocity and then update
                v_prev = self.velocities[i].copy()
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad

                # Nesterov: θ = θ - μ * v_prev + (1 + μ) * v_current
                # Simplified: θ = θ + v + μ * (v - v_prev)
                param += (
                    -self.momentum * v_prev
                    + (1 + self.momentum) * self.velocities[i]
                )

    def zero_grad(self) -> None:
        """Reset velocities (usually not needed)."""
        pass


class AdaGrad:
    """
    AdaGrad (Adaptive Gradient) optimizer.

    Formula:
        G = G + (∇L(θ))²
        θ = θ - lr * ∇L(θ) / (√G + ε)

    AdaGrad adapts learning rate per parameter, performing larger
    updates for infrequent parameters and smaller for frequent ones.

    Args:
        lr: Learning rate (default: 0.01)
        eps: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = AdaGrad(lr=0.01)

    Notes:
        - Good for sparse data (NLP, recommendation systems)
        - Learning rate decreases monotonically
        - Can stop learning prematurely (fixed by RMSprop/Adam)
    """

    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        """Initialize AdaGrad optimizer."""
        self.lr = lr
        self.eps = eps
        self.accumulated_sq_grad: Dict[int, np.ndarray] = {}

    def step(self, params: Params) -> None:
        """
        Update parameters using AdaGrad.

        Args:
            params: List of (parameter, gradient) tuples
        """
        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize accumulated squared gradient if needed
                if i not in self.accumulated_sq_grad:
                    self.accumulated_sq_grad[i] = np.zeros_like(param)

                # Accumulate squared gradient
                self.accumulated_sq_grad[i] += grad**2

                # Update with adaptive learning rate
                adaptive_lr = self.lr / (
                    np.sqrt(self.accumulated_sq_grad[i]) + self.eps
                )
                param -= adaptive_lr * grad

    def zero_grad(self) -> None:
        """No internal state to reset."""
        pass


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation) optimizer.

    Formula:
        G = β * G + (1 - β) * (∇L(θ))²
        θ = θ - lr * ∇L(θ) / (√G + ε)

    RMSprop fixes AdaGrad's aggressive learning rate decay by using
    an exponentially weighted moving average of squared gradients.

    Args:
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = RMSprop(lr=0.01, alpha=0.99)

    Notes:
        - Works well for non-stationary objectives
        - Popular choice for RNNs
        - Learning rate doesn't decrease monotonically
    """

    def __init__(self, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8):
        """Initialize RMSprop optimizer."""
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.accumulated_sq_grad: Dict[int, np.ndarray] = {}

    def step(self, params: Params) -> None:
        """
        Update parameters using RMSprop.

        Args:
            params: List of (parameter, gradient) tuples
        """
        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize accumulated squared gradient if needed
                if i not in self.accumulated_sq_grad:
                    self.accumulated_sq_grad[i] = np.zeros_like(param)

                # Exponential moving average of squared gradients
                self.accumulated_sq_grad[i] = (
                    self.alpha * self.accumulated_sq_grad[i]
                    + (1 - self.alpha) * grad**2
                )

                # Update with adaptive learning rate
                adaptive_lr = self.lr / (
                    np.sqrt(self.accumulated_sq_grad[i]) + self.eps
                )
                param -= adaptive_lr * grad

    def zero_grad(self) -> None:
        """No internal state to reset."""
        pass


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Formula:
        m = β₁ * m + (1 - β₁) * ∇L(θ)          (first moment)
        v = β₂ * v + (1 - β₂) * (∇L(θ))²       (second moment)
        m̂ = m / (1 - β₁^t)                     (bias correction)
        v̂ = v / (1 - β₂^t)                     (bias correction)
        θ = θ - lr * m̂ / (√v̂ + ε)

    Adam combines momentum (first moment) with RMSprop (second moment)
    and adds bias correction for better convergence early in training.

    Args:
        lr: Learning rate (default: 0.001)
        betas: Coefficients for (momentum, RMSprop) (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = Adam(lr=0.001, betas=(0.9, 0.999))

    Notes:
        - Most popular optimizer for deep learning
        - Works well out-of-the-box for most problems
        - Default hyperparameters rarely need tuning
        - AdamW variant (weight decay fix) preferred for Transformers

    References:
        - Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """Initialize Adam optimizer."""
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}  # First moment
        self.v: Dict[int, np.ndarray] = {}  # Second moment
        self.t: int = 0  # Time step for bias correction

    def step(self, params: Params) -> None:
        """
        Update parameters using Adam.

        Args:
            params: List of (parameter, gradient) tuples
        """
        self.t += 1

        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize moments if needed
                if i not in self.m:
                    self.m[i] = np.zeros_like(param)
                    self.v[i] = np.zeros_like(param)

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1**self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2**self.t)

                # Update parameters
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset time step (usually not needed)."""
        pass


class AdamW:
    """
    AdamW (Adam with Decoupled Weight Decay) optimizer.

    Formula:
        Same as Adam, but with decoupled weight decay:
        θ = θ - lr * wd * θ  (before Adam update)

    AdamW fixes the weight decay implementation in Adam, which was
    incorrectly coupled with the gradient-based update.

    Args:
        lr: Learning rate (default: 0.001)
        betas: Coefficients for (momentum, RMSprop) (default: (0.9, 0.999))
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)

    Example:
        >>> optimizer = AdamW(lr=0.001, weight_decay=0.01)

    Notes:
        - Preferred over Adam for Transformer training
        - Better generalization than Adam with L2 regularization
        - weight_decay=0.01 is a common choice

    References:
        - Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize AdamW optimizer."""
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: int = 0

    def step(self, params: Params) -> None:
        """
        Update parameters using AdamW.

        Args:
            params: List of (parameter, gradient) tuples
        """
        self.t += 1

        for i, (param, grad) in enumerate(params):
            if grad is not None:
                # Initialize moments if needed
                if i not in self.m:
                    self.m[i] = np.zeros_like(param)
                    self.v[i] = np.zeros_like(param)

                # Apply weight decay (decoupled from gradient)
                param -= self.lr * self.weight_decay * param

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

                # Compute bias-corrected estimates
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)

                # Update parameters
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset time step."""
        pass


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Any):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer instance with lr attribute
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self) -> None:
        """Update learning rate (called after each epoch or batch)."""
        self.step_count += 1

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


class StepLR(LRScheduler):
    """
    Step learning rate scheduler.

    Decays learning rate by gamma every step_size epochs.

    Formula: lr = base_lr * gamma^(epoch // step_size)

    Args:
        optimizer: Optimizer instance
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay

    Example:
        >>> optimizer = Adam(lr=0.001)
        >>> scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        >>> for epoch in range(30):
        ...     train(...)
        ...     scheduler.step()  # lr: 0.001 → 0.0001 → 0.00001
    """

    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1):
        """Initialize StepLR scheduler."""
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        """Decay learning rate every step_size epochs."""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.

    Formula: lr = base_lr * gamma^epoch

    Args:
        optimizer: Optimizer instance
        gamma: Multiplicative factor of learning rate decay

    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.9)
    """

    def __init__(self, optimizer: Any, gamma: float):
        """Initialize ExponentialLR scheduler."""
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        """Decay learning rate exponentially."""
        self.step_count += 1
        self.optimizer.lr = self.base_lr * (self.gamma**self.step_count)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.

    Formula: lr = η_min + (base_lr - η_min) * (1 + cos(π * epoch / T_max)) / 2

    Args:
        optimizer: Optimizer instance
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate (default: 0)

    Example:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    """

    def __init__(self, optimizer: Any, T_max: int, eta_min: float = 0.0):
        """Initialize CosineAnnealingLR scheduler."""
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self) -> None:
        """Update learning rate using cosine annealing."""
        self.step_count += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.step_count / self.T_max)
        ) / 2


# =============================================================================
# Optimizer Registry
# =============================================================================


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSprop,
    "adam": Adam,
    "adamw": AdamW,
}

SCHEDULERS = {
    "step": StepLR,
    "exponential": ExponentialLR,
    "cosine": CosineAnnealingLR,
}


def get_optimizer(name: str, **kwargs):
    """
    Get optimizer by name.

    Args:
        name: Name of optimizer
        **kwargs: Arguments to pass to optimizer constructor

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = get_optimizer("adam", lr=0.001)
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZERS.keys())}")
    return OPTIMIZERS[name](**kwargs)


def get_scheduler(name: str, optimizer: Any, **kwargs):
    """
    Get learning rate scheduler by name.

    Args:
        name: Name of scheduler
        optimizer: Optimizer instance
        **kwargs: Arguments to pass to scheduler constructor

    Returns:
        Scheduler instance

    Example:
        >>> optimizer = Adam(lr=0.001)
        >>> scheduler = get_scheduler("cosine", optimizer, T_max=100)
    """
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULERS.keys())}")
    return SCHEDULERS[name](optimizer, **kwargs)
