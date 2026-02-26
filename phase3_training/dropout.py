"""
Dropout implementations for neural network regularization.

This module provides various dropout techniques:
- Dropout: Standard dropout with inverted scaling
- MCDropout: Monte Carlo dropout for uncertainty estimation
- VariationalDropout: Dropout with learned dropout rates
- AlphaDropout: Dropout for SELU networks (preserves mean/variance)

References:
    - Dropout: A Simple Way to Prevent Neural Networks from Overfitting
      (Srivastava et al., 2014): https://arxiv.org/abs/1207.0580
    - MCDropout: Dropout as a Bayesian Approximation
      (Gal & Ghahramani, 2016): https://arxiv.org/abs/1506.02142
    - Alpha Dropout: Self-Normalizing Neural Networks
      (Klambauer et al., 2017): https://arxiv.org/abs/1706.02515
"""

from typing import Optional, Tuple, List
import numpy as np

# Type alias for array-like inputs
ArrayLike = np.ndarray


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    return x


class Dropout:
    """
    Standard Dropout with inverted scaling.

    During training, randomly zeros elements with probability p and
    scales remaining elements by 1/(1-p) to preserve expected value.
    During inference, passes input unchanged.

    Formula:
        Training: y = x * mask / (1 - p), where mask ~ Bernoulli(1 - p)
        Inference: y = x

    Args:
        p: Dropout probability (default: 0.5)

    Example:
        >>> dropout = Dropout(p=0.5)
        >>> dropout.train()
        >>> x = np.ones((10, 20))
        >>> y = dropout.forward(x)  # ~50% of elements zeroed
        >>> dropout.eval()
        >>> y = dropout.forward(x)  # unchanged
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self._training = True
        self._mask: Optional[np.ndarray] = None

    def train(self) -> None:
        """Set the layer to training mode."""
        self._training = True

    def eval(self) -> None:
        """Set the layer to evaluation mode."""
        self._training = False

    def forward(self, x: ArrayLike) -> np.ndarray:
        """
        Forward pass of dropout.

        Args:
            x: Input array of any shape

        Returns:
            Output array of same shape as input
        """
        x = _ensure_array(x)

        if not self._training or self.p == 0:
            self._mask = np.ones_like(x)
            return x

        # Inverted dropout: scale during training, not inference
        # This preserves expected value: E[y] = x
        keep_prob = 1 - self.p
        self._mask = (np.random.random(x.shape) < keep_prob).astype(x.dtype)
        return x * self._mask / keep_prob

    def backward(self, grad_output: ArrayLike) -> np.ndarray:
        """
        Backward pass of dropout.

        Args:
            grad_output: Gradient from downstream layer

        Returns:
            Gradient with respect to input
        """
        grad_output = _ensure_array(grad_output)

        if self._mask is None:
            raise RuntimeError("Must call forward before backward")

        if not self._training or self.p == 0:
            return grad_output

        # Gradient flows only through kept elements
        keep_prob = 1 - self.p
        return grad_output * self._mask / keep_prob


class MCDropout(Dropout):
    """
    Monte Carlo Dropout for uncertainty estimation.

    At inference time, keeps dropout enabled and runs multiple forward
    passes to estimate predictive uncertainty. The variance across
    predictions quantifies model uncertainty.

    This implements dropout as a Bayesian approximation, treating
    the network weights as random variables.

    Args:
        p: Dropout probability (default: 0.5)
        n_samples: Number of MC samples for prediction (default: 10)

    Example:
        >>> mc_dropout = MCDropout(p=0.3, n_samples=10)
        >>> mc_dropout.train()
        >>> x = np.random.randn(100, 50)
        >>> y = mc_dropout.forward(x)
        >>> # At inference, keep dropout enabled
        >>> mc_dropout.mc_inference = True
        >>> predictions = mc_dropout.predict(x)  # (n_samples, batch, features)
        >>> mean_pred = predictions.mean(axis=0)
        >>> uncertainty = predictions.var(axis=0)
    """

    def __init__(self, p: float = 0.5, n_samples: int = 10):
        super().__init__(p)
        self.n_samples = n_samples
        self.mc_inference = False

    def predict(self, x: ArrayLike) -> np.ndarray:
        """
        Run multiple forward passes with dropout enabled.

        Args:
            x: Input array of shape (batch, features)

        Returns:
            Predictions of shape (n_samples, batch, features)
        """
        x = _ensure_array(x)

        # Force training mode for MC sampling
        original_mode = self._training
        self._training = True

        predictions = []
        for _ in range(self.n_samples):
            pred = self.forward(x.copy())
            predictions.append(pred)

        # Restore original mode
        self._training = original_mode

        return np.stack(predictions, axis=0)


class VariationalDropout:
    """
    Variational Dropout with learned dropout rates.

    Instead of a fixed dropout probability, the dropout rate is
    learned during training. This allows the network to learn
    optimal regularization strength for each unit.

    Uses the local reparameterization trick for efficient gradient
    computation through the stochastic dropout mask.

    Formula:
        During training:
            y = x * (1 + sqrt(p/(1-p)) * epsilon), epsilon ~ N(0, 1)
        This is equivalent to Bernoulli dropout in expectation.

    Args:
        initial_p: Initial dropout probability (default: 0.5)

    References:
        - Variational Dropout and the Local Reparameterization Trick
          (Kingma et al., 2015): https://arxiv.org/abs/1506.02557
    """

    def __init__(self, initial_p: float = 0.5):
        if not 0 < initial_p < 1:
            raise ValueError(f"Initial p must be in (0, 1), got {initial_p}")
        self._logit_p = np.log(initial_p / (1 - initial_p))
        self._training = True
        self._noise: Optional[np.ndarray] = None

    @property
    def p(self) -> float:
        """Current dropout probability (computed from logit)."""
        return 1 / (1 + np.exp(-self._logit_p))

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def forward(self, x: ArrayLike) -> np.ndarray:
        """
        Forward pass with variational dropout.

        Args:
            x: Input array of any shape

        Returns:
            Output array of same shape
        """
        x = _ensure_array(x)

        if not self._training:
            return x

        # Local reparameterization trick
        p = self.p
        if p < 1e-6:  # Avoid numerical issues
            return x

        std = np.sqrt(p / (1 - p))
        self._noise = np.random.randn(*x.shape)
        return x * (1 + std * self._noise)

    def backward(self, grad_output: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Tuple of (gradient w.r.t. input, gradient w.r.t. logit_p)
        """
        grad_output = _ensure_array(grad_output)

        if self._noise is None:
            raise RuntimeError("Must call forward before backward")

        if not self._training:
            return grad_output, np.zeros(1)

        p = self.p
        if p < 1e-6:
            return grad_output, np.zeros(1)

        std = np.sqrt(p / (1 - p))
        grad_input = grad_output * (1 + std * self._noise)

        # Gradient w.r.t. logit_p
        # d(std)/d(logit_p) = d(sqrt(p/(1-p)))/d(logit_p)
        # Using p = sigmoid(logit_p), dp/d(logit_p) = p(1-p)
        grad_logit = grad_output * self._noise * p / (2 * std) if std > 0 else 0
        grad_logit_p = np.sum(grad_logit)

        return grad_input, np.array([grad_logit_p])


class AlphaDropout:
    """
    Alpha Dropout for SELU activation functions.

    When using SELU activations with self-normalizing neural networks,
    standard dropout breaks the self-normalizing property. Alpha dropout
    preserves the mean and variance of the inputs.

    Formula:
        For SELU: alpha ≈ 1.6733, scale ≈ 1.0507
        Dropped elements are set to -alpha * scale ≈ -1.7581
        Kept elements are scaled to preserve mean and variance

    Args:
        p: Dropout probability (default: 0.5)
        alpha: SELU alpha parameter (default: 1.6732632423543772)
        scale: SELU scale parameter (default: 1.0507009873554805)

    References:
        - Self-Normalizing Neural Networks (Klambauer et al., 2017)
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 1.6732632423543772,
        scale: float = 1.0507009873554805,
    ):
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.alpha = alpha
        self.scale = scale
        self._training = True
        self._mask: Optional[np.ndarray] = None

        # Precompute alpha prime for preserving self-normalizing property
        # a' = -scale * alpha
        self.alpha_p = -scale * alpha

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def forward(self, x: ArrayLike) -> np.ndarray:
        """
        Forward pass with alpha dropout.

        Args:
            x: Input array of any shape

        Returns:
            Output array of same shape
        """
        x = _ensure_array(x)

        if not self._training or self.p == 0:
            self._mask = np.ones_like(x)
            return x

        keep_prob = 1 - self.p

        # Bernoulli mask for which elements to keep
        self._mask = (np.random.random(x.shape) < keep_prob).astype(x.dtype)

        # For kept elements: multiply by factor to preserve variance
        # For dropped elements: set to alpha_p
        # This preserves E[x] and Var[x] under SELU

        # Calculate the affine transformation parameters
        # To preserve mean and variance:
        # a = (1 - p)^(-0.5) * (1 + p * alpha^2 * scale^2)^(-0.5)
        # b = -a * (1 - p) * alpha * scale

        a = np.power(keep_prob * (1 + self.p * self.alpha**2 * self.scale**2), -0.5)
        b = -a * keep_prob * self.alpha_p

        # Apply transformation
        return a * (x * self._mask + self.alpha_p * (1 - self._mask)) + b

    def backward(self, grad_output: ArrayLike) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Gradient w.r.t. input
        """
        grad_output = _ensure_array(grad_output)

        if self._mask is None:
            raise RuntimeError("Must call forward before backward")

        if not self._training or self.p == 0:
            return grad_output

        keep_prob = 1 - self.p
        a = np.power(keep_prob * (1 + self.p * self.alpha**2 * self.scale**2), -0.5)

        # Gradient only flows through kept elements, scaled by a
        return grad_output * a * self._mask


class SpatialDropout:
    """
    Spatial Dropout for convolutional networks.

    Instead of dropping individual elements, drops entire feature maps
    (channels). This is more effective for CNNs as neighboring pixels
    are highly correlated.

    Args:
        p: Dropout probability (default: 0.5)

    Example:
        >>> dropout = SpatialDropout(p=0.3)
        >>> x = np.random.randn(32, 3, 224, 224)  # (batch, channels, H, W)
        >>> y = dropout.forward(x)  # Entire channels dropped
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self._training = True
        self._mask: Optional[np.ndarray] = None

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def forward(self, x: ArrayLike) -> np.ndarray:
        """
        Forward pass with spatial dropout.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor with dropped channels
        """
        x = _ensure_array(x)

        if not self._training or self.p == 0:
            self._mask = np.ones(x.shape[1])  # One mask per channel
            return x

        if x.ndim != 4:
            raise ValueError(
                f"SpatialDropout expects 4D input (batch, channel, height, width), "
                f"got {x.ndim}D"
            )

        batch_size, channels, height, width = x.shape
        keep_prob = 1 - self.p

        # Create mask for channels only
        self._mask = (np.random.random(channels) < keep_prob).astype(x.dtype)

        # Broadcast mask to spatial dimensions
        mask = self._mask.reshape(1, channels, 1, 1)

        return x * mask / keep_prob

    def backward(self, grad_output: ArrayLike) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Gradient w.r.t. input
        """
        grad_output = _ensure_array(grad_output)

        if self._mask is None:
            raise RuntimeError("Must call forward before backward")

        if not self._training or self.p == 0:
            return grad_output

        keep_prob = 1 - self.p
        mask = self._mask.reshape(1, -1, 1, 1)
        return grad_output * mask / keep_prob


class DropConnect:
    """
    DropConnect: Drop weights instead of activations.

    Instead of dropping activations, DropConnect randomly zeros
    individual weights in the weight matrix. This is a more aggressive
    regularization than dropout.

    Args:
        p: Probability of dropping each weight (default: 0.5)

    References:
        - Regularization of Neural Networks using DropConnect
          (Wan et al., 2013)
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p < 1:
            raise ValueError(f"Drop probability must be in [0, 1), got {p}")
        self.p = p
        self._training = True
        self._weight_mask: Optional[np.ndarray] = None

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def forward(self, x: ArrayLike, weight: ArrayLike, bias: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Forward pass with DropConnect.

        Args:
            x: Input array of shape (batch, in_features)
            weight: Weight matrix of shape (out_features, in_features)
            bias: Optional bias of shape (out_features,)

        Returns:
            Output of shape (batch, out_features)
        """
        x = _ensure_array(x)
        weight = _ensure_array(weight)

        if not self._training or self.p == 0:
            self._weight_mask = np.ones_like(weight)
            output = x @ weight.T
            if bias is not None:
                output = output + bias
            return output

        keep_prob = 1 - self.p

        # Create mask for weights
        self._weight_mask = (np.random.random(weight.shape) < keep_prob).astype(weight.dtype)

        # Apply mask and scale
        masked_weight = weight * self._weight_mask / keep_prob

        output = x @ masked_weight.T
        if bias is not None:
            output = output + bias

        return output

    def backward(
        self, grad_output: ArrayLike, x: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Backward pass.

        Args:
            grad_output: Gradient from downstream (batch, out_features)
            x: Original input (batch, in_features)

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        grad_output = _ensure_array(grad_output)
        x = _ensure_array(x)

        if self._weight_mask is None:
            raise RuntimeError("Must call forward before backward")

        keep_prob = 1 - self.p

        # Gradient w.r.t. input
        grad_input = grad_output @ (self._weight_mask / keep_prob)

        # Gradient w.r.t. weight (masked)
        grad_weight = grad_output.T @ x * self._weight_mask / keep_prob

        # Gradient w.r.t. bias
        grad_bias = grad_output.sum(axis=0)

        return grad_input, grad_weight, grad_bias


def compute_mc_uncertainty(
    model_outputs: ArrayLike,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty metrics from MC Dropout samples.

    Args:
        model_outputs: Array of shape (n_samples, batch, ...) or similar
        axis: Axis along which to compute statistics (default: 0, the sample axis)

    Returns:
        Tuple of (mean, variance, entropy) where:
        - mean: Mean prediction across samples
        - variance: Variance across samples (uncertainty)
        - entropy: Predictive entropy (for classification)
    """
    model_outputs = _ensure_array(model_outputs)

    mean = np.mean(model_outputs, axis=axis)
    variance = np.var(model_outputs, axis=axis)

    # For classification (probabilities), compute predictive entropy
    # H[y|x,D] = -sum_c p(c|x,D) * log p(c|x,D)
    # where p(c|x,D) is the mean probability across samples
    # Only compute entropy for valid probability distributions (positive values summing to ~1)
    eps = 1e-10
    if mean.min() >= 0:  # Looks like probabilities
        # Normalize to ensure valid probability distribution
        mean_normalized = mean / (mean.sum(axis=-1, keepdims=True) + eps)
        entropy = -np.sum(mean_normalized * np.log(mean_normalized + eps), axis=-1)
    else:
        # For non-probability outputs, use variance as uncertainty measure
        entropy = np.mean(variance, axis=-1)

    return mean, variance, entropy


# Registry of dropout functions
DROPOUT_FUNCTIONS = {
    "dropout": Dropout,
    "mc_dropout": MCDropout,
    "variational_dropout": VariationalDropout,
    "alpha_dropout": AlphaDropout,
    "spatial_dropout": SpatialDropout,
    "drop_connect": DropConnect,
}


def get_dropout(name: str, **kwargs):
    """
    Get dropout layer by name.

    Args:
        name: Name of dropout type
        **kwargs: Arguments to pass to constructor

    Returns:
        Dropout layer instance

    Raises:
        ValueError: If unknown dropout type
    """
    name_lower = name.lower()
    if name_lower not in DROPOUT_FUNCTIONS:
        available = ", ".join(DROPOUT_FUNCTIONS.keys())
        raise ValueError(f"Unknown dropout type '{name}'. Available: {available}")
    return DROPOUT_FUNCTIONS[name_lower](**kwargs)
