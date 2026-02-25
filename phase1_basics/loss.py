"""
Neural network loss functions.

This module provides implementations of 5 common loss functions
with both forward and backward (gradient) computations using NumPy.

Functions:
    MSELoss: Mean Squared Error for regression
    CrossEntropyLoss: Cross-entropy for multi-class classification
    FocalLoss: Focal loss for class imbalance
    LabelSmoothingLoss: Label smoothing for regularization
    TripletLoss: Triplet loss for metric learning

Each class follows the pattern:
    - forward(pred, target) -> scalar loss
    - backward() -> gradient w.r.t. predictions

References:
    - Focal Loss: https://arxiv.org/abs/1708.02002
    - Label Smoothing: https://arxiv.org/abs/1512.00567
    - Triplet Loss: https://arxiv.org/abs/1503.03832
"""

import math
from typing import Union, Optional, Tuple

import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, float, int]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Convert input to numpy array if not already."""
    return np.asarray(x, dtype=np.float64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log-softmax.

    Formula: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

    Args:
        x: Input array
        axis: Axis along which to compute

    Returns:
        Log-softmax values
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


# =============================================================================
# MSE Loss (Mean Squared Error)
# =============================================================================


class MSELoss:
    """
    Mean Squared Error loss for regression.

    Formula: L = mean((pred - target)^2)

    Derivative: dL/d(pred) = 2 * (pred - target) / n

    Attributes:
        pred: Cached predictions from forward pass
        target: Cached targets from forward pass

    Example:
        >>> loss_fn = MSELoss()
        >>> pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> target = np.array([[1.5, 2.5], [2.5, 3.5]])
        >>> loss = loss_fn.forward(pred, target)
        >>> grad = loss_fn.backward()
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize MSE loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        self.reduction = reduction
        self.pred: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute MSE loss.

        Args:
            pred: Predicted values, shape (batch_size, *) or (*)
            target: Target values, same shape as pred

        Returns:
            Scalar loss value (or array if reduction='none')
        """
        self.pred = _ensure_array(pred)
        self.target = _ensure_array(target)

        diff = self.pred - self.target
        squared = diff**2

        if self.reduction == "mean":
            return float(np.mean(squared))
        elif self.reduction == "sum":
            return float(np.sum(squared))
        else:  # 'none'
            return squared

    def backward(self) -> np.ndarray:
        """
        Compute gradient of MSE loss w.r.t. predictions.

        Returns:
            Gradient array of same shape as predictions
        """
        if self.pred is None or self.target is None:
            raise RuntimeError("Must call forward() before backward()")

        diff = self.pred - self.target

        if self.reduction == "mean":
            n = self.pred.size
            return 2.0 * diff / n
        elif self.reduction == "sum":
            return 2.0 * diff
        else:  # 'none'
            return 2.0 * diff


# =============================================================================
# Cross-Entropy Loss
# =============================================================================


class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.

    Combines LogSoftmax and NLLLoss for numerical stability.

    Formula: L = -sum(target * log(softmax(pred)))

    For hard labels (class indices):
        L = -log(softmax(pred)[target_class])

    Derivative (with softmax):
        dL/d(pred) = softmax(pred) - one_hot(target)

    Attributes:
        probs: Cached softmax probabilities
        target: Cached targets (indices or one-hot)

    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        >>> targets = np.array([2, 1])  # class indices
        >>> loss = loss_fn.forward(logits, targets)
        >>> grad = loss_fn.backward()
    """

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        weight: Optional[np.ndarray] = None,
    ):
        """
        Initialize cross-entropy loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            weight: Class weights of shape (num_classes,)
        """
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.logits: Optional[np.ndarray] = None
        self.probs: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.target_one_hot: Optional[np.ndarray] = None
        self.num_classes: int = 0

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Unnormalized predictions, shape (batch_size, num_classes)
            target: Class indices shape (batch_size,) or one-hot (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        self.logits = _ensure_array(logits)
        batch_size = self.logits.shape[0]
        self.num_classes = self.logits.shape[1]

        # Convert to one-hot if needed
        if target.ndim == 1:
            # Class indices -> one-hot
            self.target = target.astype(np.int64)
            self.target_one_hot = np.zeros((batch_size, self.num_classes))
            self.target_one_hot[np.arange(batch_size), self.target] = 1.0
        else:
            # Already one-hot
            self.target_one_hot = _ensure_array(target)
            self.target = np.argmax(self.target_one_hot, axis=1).astype(np.int64)

        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_target = (1 - self.label_smoothing) * self.target_one_hot
            smooth_target += self.label_smoothing / self.num_classes
            self.target_one_hot = smooth_target

        # Compute log-softmax for numerical stability
        log_probs = _log_softmax(self.logits, axis=-1)
        self.probs = _softmax(self.logits, axis=-1)

        # Cross-entropy: -sum(target * log_softmax)
        loss = -np.sum(self.target_one_hot * log_probs, axis=-1)

        # Apply class weights
        if self.weight is not None:
            weight = np.asarray(self.weight)
            sample_weights = weight[self.target]
            loss = loss * sample_weights

        if self.reduction == "mean":
            return float(np.mean(loss))
        elif self.reduction == "sum":
            return float(np.sum(loss))
        else:  # 'none'
            return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t. logits.

        Returns:
            Gradient array of shape (batch_size, num_classes)
        """
        if self.probs is None or self.target_one_hot is None:
            raise RuntimeError("Must call forward() before backward()")

        batch_size = self.logits.shape[0]

        # Gradient: softmax(logits) - target_one_hot
        grad = self.probs - self.target_one_hot

        # Apply class weights
        if self.weight is not None:
            weight = np.asarray(self.weight)
            sample_weights = weight[self.target]
            grad = grad * sample_weights[:, np.newaxis]

        if self.reduction == "mean":
            return grad / batch_size
        elif self.reduction == "sum":
            return grad
        else:  # 'none'
            return grad


# =============================================================================
# Focal Loss
# =============================================================================


class FocalLoss:
    """
    Focal loss for handling class imbalance.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t = p if y=1, else 1-p

    Attributes:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for class balance

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        >>> targets = np.array([2, 0])
        >>> loss = loss_fn.forward(logits, targets)
        >>> grad = loss_fn.backward()

    References:
        - Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Union[float, np.ndarray] = 0.25,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter (default: 2.0)
            alpha: Class weight(s). Can be scalar or array of shape (num_classes,)
            reduction: 'mean', 'sum', or 'none'
        """
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.logits: Optional[np.ndarray] = None
        self.probs: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.target_one_hot: Optional[np.ndarray] = None
        self.num_classes: int = 0
        self.pt: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Compute focal loss.

        Args:
            logits: Unnormalized predictions, shape (batch_size, num_classes)
            target: Class indices shape (batch_size,)

        Returns:
            Scalar loss value
        """
        self.logits = _ensure_array(logits)
        batch_size = self.logits.shape[0]
        self.num_classes = self.logits.shape[1]

        # Convert to one-hot
        if target.ndim == 1:
            self.target = target.astype(np.int64)
            self.target_one_hot = np.zeros((batch_size, self.num_classes))
            self.target_one_hot[np.arange(batch_size), self.target] = 1.0
        else:
            self.target_one_hot = _ensure_array(target)
            self.target = np.argmax(self.target_one_hot, axis=1).astype(np.int64)

        # Compute softmax probabilities
        self.probs = _softmax(self.logits, axis=-1)

        # p_t: probability of the true class
        self.pt = np.sum(self.probs * self.target_one_hot, axis=-1)

        # Clip for numerical stability
        self.pt = np.clip(self.pt, 1e-8, 1.0 - 1e-8)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - self.pt) ** self.gamma

        # Cross-entropy term
        log_pt = np.log(self.pt)
        ce_loss = -log_pt

        # Alpha weighting
        if isinstance(self.alpha, np.ndarray):
            alpha_t = self.alpha[self.target]
        else:
            alpha_t = self.alpha

        # Final focal loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return float(np.mean(loss))
        elif self.reduction == "sum":
            return float(np.sum(loss))
        else:  # 'none'
            return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of focal loss w.r.t. logits.

        The gradient derivation:
        FL = -α * (1-p_t)^γ * log(p_t)

        For class k in the softmax:
        - If k is the true class (y_k=1): p_t = p_k
        - If k is not true class: p_t still depends on p_k via softmax

        Using chain rule through softmax:
        d(FL)/d(z_k) = sum_j d(FL)/d(p_j) * d(p_j)/d(z_k)

        This gives us two cases per sample:
        - For true class: grad_k = α * (1-p_t)^γ * (p_k - 1 + γ*p_k*log(p_t)/(1-p_t))
        - For other classes: grad_k = α * p_k * (1-p_t)^(γ-1) * ((1-p_t) - γ*log(p_t)*p_t)

        Simplified implementation using the standard focal loss gradient formula:
        d(FL)/d(z) = α * (p - y) * (1-p_t)^γ * (1 + γ*log(p_t)/(1-p_t))

        Returns:
            Gradient array of shape (batch_size, num_classes)
        """
        if self.probs is None or self.target_one_hot is None or self.pt is None:
            raise RuntimeError("Must call forward() before backward()")

        batch_size = self.logits.shape[0]

        # Get alpha for each sample
        if isinstance(self.alpha, np.ndarray):
            alpha_t = self.alpha[self.target]
        else:
            alpha_t = np.full(batch_size, self.alpha)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - self.pt) ** self.gamma

        # Log of p_t (clipped for stability)
        log_pt = np.log(np.clip(self.pt, 1e-8, 1.0))

        # The focal modulation factor
        # For true class: the derivative includes an extra term
        # For non-true classes: simpler derivative

        # Using the standard formula:
        # d(FL)/d(z_k) = α * [(1-p_t)^γ * (p_k * (-1 if k is true class else 0) + sum_term)]

        # A cleaner approach: compute gradient component by component
        # FL = -α * (1-p_t)^γ * log(p_t)
        # Let f = (1-p_t)^γ, g = log(p_t)
        # FL = -α * f * g
        # d(FL)/d(p_k) = -α * (df/d(p_k) * g + f * dg/d(p_k))

        # df/d(p_k) = -γ * (1-p_t)^(γ-1) * d(p_t)/d(p_k) = -γ * (1-p_t)^(γ-1) * y_k
        # dg/d(p_k) = (1/p_t) * y_k

        # For true class (y_k=1): d(FL)/d(p_k) = -α * (-γ * (1-p_t)^(γ-1) * log(p_t) + (1-p_t)^γ / p_t)
        # For other classes (y_k=0): d(FL)/d(p_k) = 0 (directly)
        # But we need d(FL)/d(z_k), not d(FL)/d(p_k)

        # Using softmax gradient: d(p_j)/d(z_k) = p_j * (δ_jk - p_k)
        # This makes the full gradient complex. Use a verified formula instead.

        # Verified formula from focal loss implementations:
        # d(FL)/d(z_k) = α * p_k * (1 - y_k/focal_weight*p_t^(1-γ)) - α * y_k * (1 + γ*log(p_t)/(1-p_t))
        #              = α * p_k - α * y_k + α * y_k * (1 - p_t) * γ * log(p_t) / ((1-p_t) * p_t) * p_t
        #
        # Simplified: d(FL)/d(z_k) = α * (p_k - y_k) * (1-p_t)^γ * (1 + γ*log(p_t)/(1-p_t)/(1-p_t) * p_t * (p_k - y_k) / (p_k - y_k))
        #
        # Most common implementation:
        # grad_k = α * (p_k - y_k) * (1 - p_t)^γ * (1 + γ * p_t * log(p_t) / (1 - p_t))

        # Focal loss gradient derivation (verified correct)
        #
        # FL = -α * (1 - p_t)^γ * log(p_t)
        # where p_t = probability of the true class
        #
        # Using chain rule:
        # d(FL)/d(z_k) = d(FL)/d(p_t) * d(p_t)/d(z_k)
        #
        # where:
        #   d(FL)/d(p_t) = α * (1-p_t)^(γ-1) * (γ * log(p_t) - (1-p_t)/p_t)
        #   d(p_t)/d(z_k) = p_t * (δ_tk - p_k)  where t is the true class index
        #
        # Combining:
        #   d(FL)/d(z_k) = α * (1-p_t)^(γ-1) * (γ * log(p_t) - (1-p_t)/p_t) * p_t * (δ_tk - p_k)
        #                = α * (1-p_t)^(γ-1) * p_t * (γ * log(p_t) - (1-p_t)/p_t) * (δ_tk - p_k)

        one_minus_pt = np.clip(1.0 - self.pt, 1e-8, 1.0 - 1e-8)
        log_pt = np.log(np.clip(self.pt, 1e-8, 1.0))

        # d(FL)/d(p_t) = α * (1-p_t)^(γ-1) * (γ * log(p_t) - (1-p_t)/p_t)
        dFL_dpt = alpha_t * (one_minus_pt ** (self.gamma - 1)) * (
            self.gamma * log_pt - one_minus_pt / self.pt
        )

        # Compute gradient for each class k
        # d(p_t)/d(z_k) = p_t * (δ_tk - p_k)
        # d(FL)/d(z_k) = dFL_dpt * p_t * (δ_tk - p_k)
        #
        # For true class k=t: δ_tk = 1, so d(p_t)/d(z_k) = p_t * (1 - p_k) = p_t * (1 - p_t)
        # For other classes k≠t: δ_tk = 0, so d(p_t)/d(z_k) = p_t * (0 - p_k) = -p_t * p_k

        # Compute using broadcasting:
        # For each sample i with true class t_i:
        #   grad[i, k] = dFL_dpt[i] * p_t[i] * (δ(t_i, k) - probs[i, k])
        #             = dFL_dpt[i] * p_t[i] * (target_one_hot[i, k] - probs[i, k])

        grad = dFL_dpt[:, np.newaxis] * self.pt[:, np.newaxis] * (
            self.target_one_hot - self.probs
        )

        if self.reduction == "mean":
            return grad / batch_size
        elif self.reduction == "sum":
            return grad
        else:  # 'none'
            return grad

        if self.reduction == "mean":
            return grad / batch_size
        elif self.reduction == "sum":
            return grad
        else:  # 'none'
            return grad


# =============================================================================
# Label Smoothing Loss (Standalone)
# =============================================================================


class LabelSmoothingLoss:
    """
    Label smoothing loss for regularization.

    Replaces hard one-hot labels with smoothed versions:
        y_smooth = (1 - epsilon) * y_hard + epsilon / num_classes

    This prevents overconfidence and acts as regularization.

    Formula:
        y_smooth[k] = 1 - epsilon + epsilon/K  (for true class)
        y_smooth[k] = epsilon / K              (for other classes)

    Example:
        >>> loss_fn = LabelSmoothingLoss(epsilon=0.1)
        >>> logits = np.array([[1.0, 2.0, 3.0]])
        >>> target = np.array([2])  # true class is 2
        >>> loss = loss_fn.forward(logits, target)
        >>> # Smoothed target: [0.033, 0.033, 0.933]

    References:
        - Rethinking the Inception Architecture for Computer Vision: https://arxiv.org/abs/1512.00567
    """

    def __init__(self, epsilon: float = 0.1, reduction: str = "mean"):
        """
        Initialize label smoothing loss.

        Args:
            epsilon: Smoothing factor (default: 0.1)
            reduction: 'mean', 'sum', or 'none'
        """
        self.epsilon = epsilon
        self.reduction = reduction
        self.logits: Optional[np.ndarray] = None
        self.probs: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.smooth_target: Optional[np.ndarray] = None
        self.num_classes: int = 0

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        """
        Compute label smoothing loss.

        Args:
            logits: Unnormalized predictions, shape (batch_size, num_classes)
            target: Class indices, shape (batch_size,)

        Returns:
            Scalar loss value
        """
        self.logits = _ensure_array(logits)
        batch_size = self.logits.shape[0]
        self.num_classes = self.logits.shape[1]

        self.target = target.astype(np.int64)

        # Create smoothed labels
        self.smooth_target = np.full(
            (batch_size, self.num_classes), self.epsilon / self.num_classes
        )
        self.smooth_target[np.arange(batch_size), self.target] = (
            1.0 - self.epsilon + self.epsilon / self.num_classes
        )

        # Compute log-softmax
        log_probs = _log_softmax(self.logits, axis=-1)
        self.probs = _softmax(self.logits, axis=-1)

        # KL divergence style loss: -sum(smooth_target * log_softmax)
        loss = -np.sum(self.smooth_target * log_probs, axis=-1)

        if self.reduction == "mean":
            return float(np.mean(loss))
        elif self.reduction == "sum":
            return float(np.sum(loss))
        else:  # 'none'
            return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of label smoothing loss w.r.t. logits.

        Returns:
            Gradient array of shape (batch_size, num_classes)
        """
        if self.probs is None or self.smooth_target is None:
            raise RuntimeError("Must call forward() before backward()")

        batch_size = self.logits.shape[0]

        # Gradient: softmax - smooth_target
        grad = self.probs - self.smooth_target

        if self.reduction == "mean":
            return grad / batch_size
        elif self.reduction == "sum":
            return grad
        else:  # 'none'
            return grad


# =============================================================================
# Triplet Loss
# =============================================================================


class TripletLoss:
    """
    Triplet loss for metric learning.

    Learns embeddings where similar samples are closer than dissimilar ones.

    Formula: L = max(d(a, p) - d(a, n) + margin, 0)

    where:
        a = anchor sample
        p = positive sample (same class as anchor)
        n = negative sample (different class from anchor)
        d = distance function (default: Euclidean)

    Attributes:
        margin: Minimum distance difference between positive and negative pairs
        p: Distance norm (default: 2 for Euclidean)

    Example:
        >>> loss_fn = TripletLoss(margin=0.5)
        >>> anchor = np.array([[1.0, 2.0]])
        >>> positive = np.array([[1.1, 2.1]])
        >>> negative = np.array([[5.0, 5.0]])
        >>> loss = loss_fn.forward(anchor, positive, negative)
        >>> (grad_a, grad_p, grad_n) = loss_fn.backward()

    References:
        - FaceNet: A Unified Embedding for Face Recognition: https://arxiv.org/abs/1503.03832
    """

    def __init__(self, margin: float = 1.0, p: float = 2.0, reduction: str = "mean"):
        """
        Initialize triplet loss.

        Args:
            margin: Margin between positive and negative distances (default: 1.0)
            p: Distance norm (2 = Euclidean, 1 = Manhattan)
            reduction: 'mean', 'sum', or 'none'
        """
        self.margin = margin
        self.p = p
        self.reduction = reduction
        self.anchor: Optional[np.ndarray] = None
        self.positive: Optional[np.ndarray] = None
        self.negative: Optional[np.ndarray] = None
        self.d_pos: Optional[np.ndarray] = None
        self.d_neg: Optional[np.ndarray] = None
        self.loss_value: Optional[np.ndarray] = None

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance.

        Args:
            x1: First tensor, shape (batch_size, embedding_dim)
            x2: Second tensor, shape (batch_size, embedding_dim)

        Returns:
            Distance tensor, shape (batch_size,)
        """
        diff = x1 - x2
        if self.p == 2:
            return np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)
        else:
            return np.sum(np.abs(diff) ** self.p, axis=-1) ** (1.0 / self.p)

    def forward(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
    ) -> float:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings, shape (batch_size, embedding_dim)
            positive: Positive embeddings (same class), shape (batch_size, embedding_dim)
            negative: Negative embeddings (different class), shape (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        self.anchor = _ensure_array(anchor)
        self.positive = _ensure_array(positive)
        self.negative = _ensure_array(negative)

        # Compute distances
        self.d_pos = self._compute_distance(self.anchor, self.positive)
        self.d_neg = self._compute_distance(self.anchor, self.negative)

        # Triplet loss: max(d_pos - d_neg + margin, 0)
        self.loss_value = np.maximum(self.d_pos - self.d_neg + self.margin, 0.0)

        if self.reduction == "mean":
            return float(np.mean(self.loss_value))
        elif self.reduction == "sum":
            return float(np.sum(self.loss_value))
        else:  # 'none'
            return self.loss_value

    def backward(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients of triplet loss w.r.t. anchor, positive, and negative.

        Returns:
            Tuple of (grad_anchor, grad_positive, grad_negative)
        """
        if self.anchor is None or self.d_pos is None or self.d_neg is None:
            raise RuntimeError("Must call forward() before backward()")

        batch_size = self.anchor.shape[0]

        # Mask for active triplets (those with non-zero loss)
        active = (self.d_pos - self.d_neg + self.margin > 0).astype(np.float64)
        if self.reduction == "mean":
            active = active / batch_size

        # Gradient for Euclidean distance (p=2):
        # d(d_pos)/d(anchor) = (anchor - positive) / d_pos
        # d(d_pos)/d(positive) = (positive - anchor) / d_pos
        # d(d_neg)/d(anchor) = (anchor - negative) / d_neg
        # d(d_neg)/d(negative) = (negative - anchor) / d_neg

        if self.p == 2:
            # Gradient w.r.t. anchor: +1 for d_pos term, -1 for d_neg term
            grad_a_pos = (self.anchor - self.positive) / (
                self.d_pos[:, np.newaxis] + 1e-8
            )
            grad_a_neg = -(self.anchor - self.negative) / (
                self.d_neg[:, np.newaxis] + 1e-8
            )
            grad_anchor = active[:, np.newaxis] * (grad_a_pos + grad_a_neg)

            # Gradient w.r.t. positive: -1 * direction from anchor to positive
            grad_positive = active[:, np.newaxis] * (
                -(self.anchor - self.positive) / (self.d_pos[:, np.newaxis] + 1e-8)
            )

            # Gradient w.r.t. negative: +1 * direction from anchor to negative
            grad_negative = active[:, np.newaxis] * (
                (self.anchor - self.negative) / (self.d_neg[:, np.newaxis] + 1e-8)
            )
        else:
            # General p-norm gradient
            diff_ap = self.anchor - self.positive
            diff_an = self.anchor - self.negative

            sign_ap = np.sign(diff_ap)
            sign_an = np.sign(diff_an)

            grad_anchor = active[:, np.newaxis] * (
                sign_ap * np.abs(diff_ap) ** (self.p - 1) / (self.d_pos[:, np.newaxis] + 1e-8)
                - sign_an * np.abs(diff_an) ** (self.p - 1) / (self.d_neg[:, np.newaxis] + 1e-8)
            )
            grad_positive = active[:, np.newaxis] * (
                -sign_ap * np.abs(diff_ap) ** (self.p - 1) / (self.d_pos[:, np.newaxis] + 1e-8)
            )
            grad_negative = active[:, np.newaxis] * (
                sign_an * np.abs(diff_an) ** (self.p - 1) / (self.d_neg[:, np.newaxis] + 1e-8)
            )

        return grad_anchor, grad_positive, grad_negative


# =============================================================================
# Utility function for gradient checking
# =============================================================================


def numerical_gradient_loss(
    loss_fn,
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute numerical gradient for loss function.

    Formula: ∂L/∂x ≈ (L(x+ε) - L(x-ε)) / (2ε)

    Args:
        loss_fn: Loss function instance with forward() method
        pred: Prediction array
        target: Target array
        eps: Small perturbation for finite difference

    Returns:
        Numerical gradient array of same shape as pred
    """
    grad = np.zeros_like(pred, dtype=np.float64)
    it = np.nditer(pred, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = pred[idx]

        pred[idx] = old_val + eps
        f_plus = loss_fn.forward(pred, target)

        pred[idx] = old_val - eps
        f_minus = loss_fn.forward(pred, target)

        pred[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()

    return grad


def numerical_gradient_triplet(
    loss_fn: TripletLoss,
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute numerical gradient for triplet loss.

    Args:
        loss_fn: TripletLoss instance
        anchor: Anchor embeddings
        positive: Positive embeddings
        negative: Negative embeddings
        eps: Small perturbation

    Returns:
        Tuple of (grad_anchor, grad_positive, grad_negative)
    """
    # Gradient w.r.t. anchor
    grad_a = np.zeros_like(anchor, dtype=np.float64)
    for idx in np.ndindex(anchor.shape):
        old = anchor[idx]
        anchor[idx] = old + eps
        f_plus = loss_fn.forward(anchor, positive, negative)
        anchor[idx] = old - eps
        f_minus = loss_fn.forward(anchor, positive, negative)
        anchor[idx] = old
        grad_a[idx] = (f_plus - f_minus) / (2.0 * eps)

    # Gradient w.r.t. positive
    grad_p = np.zeros_like(positive, dtype=np.float64)
    for idx in np.ndindex(positive.shape):
        old = positive[idx]
        positive[idx] = old + eps
        f_plus = loss_fn.forward(anchor, positive, negative)
        positive[idx] = old - eps
        f_minus = loss_fn.forward(anchor, positive, negative)
        positive[idx] = old
        grad_p[idx] = (f_plus - f_minus) / (2.0 * eps)

    # Gradient w.r.t. negative
    grad_n = np.zeros_like(negative, dtype=np.float64)
    for idx in np.ndindex(negative.shape):
        old = negative[idx]
        negative[idx] = old + eps
        f_plus = loss_fn.forward(anchor, positive, negative)
        negative[idx] = old - eps
        f_minus = loss_fn.forward(anchor, positive, negative)
        negative[idx] = old
        grad_n[idx] = (f_plus - f_minus) / (2.0 * eps)

    return grad_a, grad_p, grad_n


# =============================================================================
# Loss function registry
# =============================================================================

LOSS_FUNCTIONS = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "triplet": TripletLoss,
}


def get_loss(name: str, **kwargs):
    """
    Get loss function by name.

    Args:
        name: Name of loss function
        **kwargs: Arguments to pass to loss constructor

    Returns:
        Loss function instance

    Example:
        >>> loss_fn = get_loss("cross_entropy", label_smoothing=0.1)
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss: {name}. Available: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[name](**kwargs)
