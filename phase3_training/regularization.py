"""
Regularization techniques for neural network weight penalization.

This module provides regularization methods:
- L1 Regularization (Lasso): Encourages sparsity
- L2 Regularization (Ridge/Weight Decay): Prevents large weights
- ElasticNet: Combination of L1 and L2
- MaxNorm: Constrains weight norms

These can be used standalone or integrated with optimizers.

References:
    - Regression Shrinkage and Selection via the Lasso (Tibshirani, 1996)
    - Regularization and Variable Selection via the Elastic Net (Zou & Hastie, 2005)
"""

from typing import List, Tuple, Optional, Union, Callable
import numpy as np

# Type alias
ArrayLike = np.ndarray


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    return x


class L1Regularization:
    """
    L1 (Lasso) Regularization.

    Adds the sum of absolute values of weights to the loss:
        L1 = lambda * sum(|w|)

    Properties:
    - Encourages sparsity (drives weights to exactly zero)
    - Good for feature selection
    - Robust to outliers

    Args:
        lambda_: Regularization strength (default: 0.01)

    Example:
        >>> reg = L1Regularization(lambda_=0.01)
        >>> weights = [np.random.randn(10, 20), np.random.randn(20, 5)]
        >>> loss = reg.loss(weights)
        >>> grads = reg.gradient(weights)
    """

    def __init__(self, lambda_: float = 0.01):
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        self.lambda_ = lambda_

    def loss(self, weights: Union[List[ArrayLike], ArrayLike]) -> float:
        """
        Compute L1 regularization loss.

        Args:
            weights: Single weight array or list of weight arrays

        Returns:
            L1 regularization term
        """
        if isinstance(weights, np.ndarray):
            weights = [weights]

        return self.lambda_ * sum(np.sum(np.abs(w)) for w in weights)

    def gradient(self, weights: Union[List[ArrayLike], ArrayLike]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute gradient of L1 term.

        Args:
            weights: Single weight array or list of weight arrays

        Returns:
            Gradient(s) - subgradient at 0 is 0
        """
        if isinstance(weights, np.ndarray):
            weights = [weights]
            single = True
        else:
            single = False

        grads = []
        for w in weights:
            w = _ensure_array(w)
            # Subgradient of |w| is sign(w), with 0 at w=0
            grad = self.lambda_ * np.sign(w)
            grads.append(grad)

        return grads[0] if single else grads


class L2Regularization:
    """
    L2 (Ridge/Weight Decay) Regularization.

    Adds the sum of squared weights to the loss:
        L2 = lambda * sum(w^2) / 2

    Note: The factor of 1/2 makes the gradient simply lambda * w

    Properties:
    - Prevents any single weight from becoming too large
    - Smooth penalty (differentiable everywhere)
    - Equivalent to Gaussian prior on weights (Bayesian view)
    - Standard "weight decay" in deep learning

    Args:
        lambda_: Regularization strength (default: 0.01)

    Example:
        >>> reg = L2Regularization(lambda_=0.01)
        >>> weights = [np.random.randn(10, 20), np.random.randn(20, 5)]
        >>> loss = reg.loss(weights)
        >>> grads = reg.gradient(weights)
    """

    def __init__(self, lambda_: float = 0.01):
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        self.lambda_ = lambda_

    def loss(self, weights: Union[List[ArrayLike], ArrayLike]) -> float:
        """
        Compute L2 regularization loss.

        Args:
            weights: Single weight array or list of weight arrays

        Returns:
            L2 regularization term (sum of squared weights / 2)
        """
        if isinstance(weights, np.ndarray):
            weights = [weights]

        # Factor of 0.5 for cleaner gradient
        return 0.5 * self.lambda_ * sum(np.sum(w**2) for w in weights)

    def gradient(self, weights: Union[List[ArrayLike], ArrayLike]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute gradient of L2 term.

        Args:
            weights: Single weight array or list of weight arrays

        Returns:
            Gradient(s): lambda * w for each weight
        """
        if isinstance(weights, np.ndarray):
            weights = [weights]
            single = True
        else:
            single = False

        grads = [self.lambda_ * _ensure_array(w) for w in weights]

        return grads[0] if single else grads


class ElasticNet:
    """
    Elastic Net Regularization (L1 + L2 combined).

    Combines L1 and L2 regularization:
        ElasticNet = alpha * L1 + (1 - alpha) * L2
                   = alpha * lambda * sum(|w|) + (1 - alpha) * lambda * sum(w^2) / 2

    Properties:
    - Combines sparsity (L1) with stability (L2)
    - Good when features are correlated
    - Groups of correlated features tend to be in/out together

    Args:
        lambda_: Total regularization strength (default: 0.01)
        alpha: Mix between L1 (alpha) and L2 (1-alpha) (default: 0.5)

    Example:
        >>> reg = ElasticNet(lambda_=0.01, alpha=0.5)
        >>> weights = [np.random.randn(10, 20)]
        >>> loss = reg.loss(weights)
    """

    def __init__(self, lambda_: float = 0.01, alpha: float = 0.5):
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

        self.lambda_ = lambda_
        self.alpha = alpha
        self._l1 = L1Regularization(lambda_ * alpha)
        self._l2 = L2Regularization(lambda_ * (1 - alpha))

    def loss(self, weights: Union[List[ArrayLike], ArrayLike]) -> float:
        """Compute Elastic Net regularization loss."""
        return self._l1.loss(weights) + self._l2.loss(weights)

    def gradient(self, weights: Union[List[ArrayLike], ArrayLike]) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute gradient of Elastic Net term."""
        if isinstance(weights, np.ndarray):
            weights = [weights]
            single = True
        else:
            single = False

        l1_grads = self._l1.gradient(weights)
        l2_grads = self._l2.gradient(weights)

        if single:
            return l1_grads + l2_grads

        return [l1 + l2 for l1, l2 in zip(l1_grads, l2_grads)]


class MaxNormConstraint:
    """
    Max-Norm Constraint.

    Constrains the L2 norm of weight vectors to be <= max_value.
    After each gradient update, weights are projected back to the
    constraint ball if violated.

    Commonly used with dropout and ReLU activations.

    Args:
        max_value: Maximum allowed L2 norm (default: 2.0)
        axis: Axis along which to compute norm (default: 0)

    Example:
        >>> constraint = MaxNormConstraint(max_value=2.0)
        >>> weights = np.random.randn(100, 50)
        >>> constrained = constraint(weights)
    """

    def __init__(self, max_value: float = 2.0, axis: int = 0):
        if max_value <= 0:
            raise ValueError(f"max_value must be positive, got {max_value}")
        self.max_value = max_value
        self.axis = axis

    def __call__(self, weights: ArrayLike) -> np.ndarray:
        """
        Apply max-norm constraint.

        Args:
            weights: Weight array to constrain

        Returns:
            Constrained weights (same shape as input)
        """
        weights = _ensure_array(weights)
        norms = np.sqrt(np.sum(weights**2, axis=self.axis, keepdims=True))
        # Only scale down if norm exceeds max_value
        # desired = min(norm, max_value), so we want:
        # - If norm > max_value: scale to max_value (factor = max_value / norm)
        # - If norm <= max_value: no scaling (factor = 1)
        scale = np.where(norms > self.max_value, self.max_value / (norms + 1e-7), 1.0)
        return weights * scale

    def project(self, weights: ArrayLike) -> np.ndarray:
        """Alias for __call__."""
        return self(weights)


class L1L2Regularizer:
    """
    Combined L1 and L2 regularizer with separate strengths.

    Unlike ElasticNet which shares lambda_, this allows independent
    control of L1 and L2 strengths.

    Formula: l1 * sum(|w|) + l2 * sum(w^2) / 2

    Args:
        l1: L1 regularization factor (default: 0.01)
        l2: L2 regularization factor (default: 0.01)
    """

    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        if l1 < 0 or l2 < 0:
            raise ValueError(f"l1 and l2 must be non-negative, got l1={l1}, l2={l2}")
        self.l1 = l1
        self.l2 = l2

    def loss(self, weights: Union[List[ArrayLike], ArrayLike]) -> float:
        """Compute combined L1+L2 loss."""
        if isinstance(weights, np.ndarray):
            weights = [weights]

        l1_term = sum(np.sum(np.abs(w)) for w in weights)
        l2_term = 0.5 * sum(np.sum(w**2) for w in weights)

        return self.l1 * l1_term + self.l2 * l2_term

    def gradient(self, weights: Union[List[ArrayLike], ArrayLike]) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute combined gradient."""
        if isinstance(weights, np.ndarray):
            weights = [weights]
            single = True
        else:
            single = False

        grads = []
        for w in weights:
            w = _ensure_array(w)
            grad = self.l1 * np.sign(w) + self.l2 * w
            grads.append(grad)

        return grads[0] if single else grads


class OrthogonalRegularizer:
    """
    Orthogonal Regularization for weight matrices.

    Encourages weight matrices to be orthogonal by penalizing
    W^T W - I (for fully connected) or W W^T - I.

    Useful for:
    - RNNs to prevent vanishing/exploding gradients
    - Preventing feature redundancy

    Args:
        lambda_: Regularization strength (default: 0.01)

    References:
        - Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
          (Saxe et al., 2013)
    """

    def __init__(self, lambda_: float = 0.01):
        if lambda_ < 0:
            raise ValueError(f"Lambda must be non-negative, got {lambda_}")
        self.lambda_ = lambda_

    def loss(self, weights: ArrayLike) -> float:
        """
        Compute orthogonal regularization loss.

        Args:
            weights: Weight matrix of shape (out_features, in_features)

        Returns:
            Regularization loss
        """
        weights = _ensure_array(weights)

        # W^T W should be close to identity for orthogonal
        n = weights.shape[1]
        product = weights.T @ weights
        identity = np.eye(n)
        return 0.5 * self.lambda_ * np.sum((product - identity) ** 2)

    def gradient(self, weights: ArrayLike) -> np.ndarray:
        """
        Compute gradient of orthogonal regularization.

        Args:
            weights: Weight matrix of shape (out_features, in_features)

        Returns:
            Gradient with same shape as weights
        """
        weights = _ensure_array(weights)
        n = weights.shape[1]
        product = weights.T @ weights
        identity = np.eye(n)
        return self.lambda_ * weights @ (product - identity)


class SpectralNormConstraint:
    """
    Spectral Normalization Constraint.

    Constrains the spectral norm (largest singular value) of weight
    matrices to be <= max_value.

    Used in GANs (SN-GAN) for training stability.

    Args:
        max_value: Maximum spectral norm (default: 1.0)
        n_power_iterations: Number of power iterations for SVD (default: 1)

    References:
        - Spectral Normalization for Generative Adversarial Networks
          (Miyato et al., 2018): https://arxiv.org/abs/1802.05957
    """

    def __init__(self, max_value: float = 1.0, n_power_iterations: int = 1):
        if max_value <= 0:
            raise ValueError(f"max_value must be positive, got {max_value}")
        self.max_value = max_value
        self.n_power_iterations = n_power_iterations
        self._u: Optional[np.ndarray] = None

    def _power_iteration(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Power iteration method to estimate spectral norm.

        Args:
            w: Weight matrix

        Returns:
            Tuple of (u, v, sigma) where sigma is the spectral norm
        """
        if self._u is None:
            # Initialize random vector
            self._u = np.random.randn(w.shape[0])
            self._u = self._u / np.linalg.norm(self._u)

        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = w.T @ self._u
            v = v / (np.linalg.norm(v) + 1e-7)

            # u = W v / ||W v||
            self._u = w @ v
            self._u = self._u / (np.linalg.norm(self._u) + 1e-7)

        # sigma = u^T W v
        sigma = self._u @ w @ v

        return self._u, v, sigma

    def __call__(self, weights: ArrayLike) -> np.ndarray:
        """
        Apply spectral normalization.

        Args:
            weights: Weight matrix of shape (out_features, in_features)

        Returns:
            Normalized weights
        """
        weights = _ensure_array(weights)
        _, _, sigma = self._power_iteration(weights)

        if sigma > self.max_value:
            return weights * (self.max_value / sigma)

        return weights


def apply_weight_decay(
    params: List[ArrayLike],
    grads: List[ArrayLike],
    lr: float,
    weight_decay: float = 0.0,
    decay_type: str = "l2",
) -> List[np.ndarray]:
    """
    Apply weight decay to gradients.

    This is the standard way to implement L2 regularization in SGD:
        grad_new = grad + weight_decay * param

    Args:
        params: List of parameter arrays
        grads: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        decay_type: "l2" or "l1"

    Returns:
        Updated gradients with weight decay applied
    """
    if weight_decay == 0:
        return [g.copy() for g in grads]

    updated_grads = []
    for param, grad in zip(params, grads):
        param = _ensure_array(param)
        grad = _ensure_array(grad)

        if decay_type == "l2":
            updated_grads.append(grad + weight_decay * param)
        elif decay_type == "l1":
            updated_grads.append(grad + weight_decay * np.sign(param))
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

    return updated_grads


def compute_regularization_loss(
    weights: Union[List[ArrayLike], ArrayLike],
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
) -> float:
    """
    Compute combined regularization loss.

    Args:
        weights: Weight array(s)
        l1_lambda: L1 regularization strength
        l2_lambda: L2 regularization strength

    Returns:
        Total regularization loss
    """
    if isinstance(weights, np.ndarray):
        weights = [weights]

    loss = 0.0

    if l1_lambda > 0:
        loss += l1_lambda * sum(np.sum(np.abs(w)) for w in weights)

    if l2_lambda > 0:
        loss += 0.5 * l2_lambda * sum(np.sum(w**2) for w in weights)

    return loss


# Registry
REGULARIZATION_FUNCTIONS = {
    "l1": L1Regularization,
    "l2": L2Regularization,
    "elastic_net": ElasticNet,
    "l1l2": L1L2Regularizer,
    "max_norm": MaxNormConstraint,
    "orthogonal": OrthogonalRegularizer,
    "spectral_norm": SpectralNormConstraint,
}


def get_regularizer(name: str, **kwargs):
    """
    Get regularizer by name.

    Args:
        name: Name of regularizer
        **kwargs: Arguments to pass to constructor

    Returns:
        Regularizer instance
    """
    name_lower = name.lower().replace("-", "_")
    if name_lower not in REGULARIZATION_FUNCTIONS:
        available = ", ".join(REGULARIZATION_FUNCTIONS.keys())
        raise ValueError(f"Unknown regularizer '{name}'. Available: {available}")
    return REGULARIZATION_FUNCTIONS[name_lower](**kwargs)
