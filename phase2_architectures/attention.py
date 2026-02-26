"""
Self-Attention and Multi-Head Attention implementations from scratch.

This module provides NumPy implementations of the attention mechanism
used in Transformer architectures.

Theory:
    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Multi-Head Attention:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
        where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)

    Sinusoidal Positional Encoding:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

References:
    - Attention Is All You Need (Vaswani et al., 2017)
    - The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
"""

from typing import Tuple, Optional, Union
import math
import numpy as np

ArrayLike = Union[np.ndarray, list, float]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array with float64 dtype."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)
    return x


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# Scaled Dot-Product Attention
# =============================================================================


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        d_k: Dimension of keys (used for scaling)
        dropout: Dropout probability (not implemented in NumPy version)

    Attributes:
        cache: Cached values for backward pass
        attn_weights: Attention weights from last forward pass
    """

    def __init__(self, d_k: int, dropout: float = 0.0):
        self.d_k = d_k
        self.dropout = dropout
        self.cache = None
        self.attn_weights = None

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass for scaled dot-product attention.

        Args:
            query: Query tensor, shape (batch, heads, seq_q, d_k)
            key: Key tensor, shape (batch, heads, seq_k, d_k)
            value: Value tensor, shape (batch, heads, seq_v, d_v)
                   Note: seq_k == seq_v for self-attention
            mask: Optional mask, shape (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
                  True/1 positions are masked (not attended to)

        Returns:
            output: Attention output, shape (batch, heads, seq_q, d_v)
        """
        query = _ensure_array(query)
        key = _ensure_array(key)
        value = _ensure_array(value)

        # Compute attention scores: QK^T / sqrt(d_k)
        # query: (batch, heads, seq_q, d_k)
        # key.T: (batch, heads, d_k, seq_k)
        # scores: (batch, heads, seq_q, seq_k)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        # Apply mask (if provided)
        if mask is not None:
            # Mask positions with very negative value before softmax
            scores = np.where(mask == 1, -1e9, scores)

        # Softmax to get attention weights
        self.attn_weights = softmax(scores, axis=-1)

        # Apply attention weights to values
        # attn_weights: (batch, heads, seq_q, seq_k)
        # value: (batch, heads, seq_v, d_v)
        # output: (batch, heads, seq_q, d_v)
        output = np.matmul(self.attn_weights, value)

        # Cache for backward pass
        self.cache = (query, key, value, mask)

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for scaled dot-product attention.

        The gradient computation follows the chain rule through:
        1. d/dV (attn @ V) = attn
        2. d/d(attn) (attn @ V) = V
        3. d/d(scores) softmax(scores) = softmax Jacobian
        4. d/dQ, d/dK (QK^T) = K, Q respectively

        Args:
            grad_output: Gradient from next layer, shape (batch, heads, seq_q, d_v)

        Returns:
            grad_query: Gradient w.r.t. query, shape (batch, heads, seq_q, d_k)
            grad_key: Gradient w.r.t. key, shape (batch, heads, seq_k, d_k)
            grad_value: Gradient w.r.t. value, shape (batch, heads, seq_v, d_v)
        """
        if self.cache is None:
            raise RuntimeError("Must call forward before backward")

        query, key, value, mask = self.cache
        attn = self.attn_weights

        batch_size, num_heads, seq_q, d_k = query.shape
        _, _, seq_k, d_v = value.shape

        # Gradient w.r.t. value: d/dV (attn @ V) = attn^T @ grad_output
        # attn: (batch, heads, seq_q, seq_k)
        # grad_output: (batch, heads, seq_q, d_v)
        # grad_value: (batch, heads, seq_k, d_v)
        grad_value = np.matmul(attn.transpose(0, 1, 3, 2), grad_output)

        # Gradient w.r.t. attention weights: d/d(attn) (attn @ V) = grad_output @ V^T
        # grad_attn: (batch, heads, seq_q, seq_k)
        grad_attn = np.matmul(grad_output, value.transpose(0, 1, 3, 2))

        # Gradient through softmax: d/dx softmax(x) = softmax(x) * (grad - sum(softmax * grad))
        # This is the Jacobian-vector product for softmax
        # grad_scores: (batch, heads, seq_q, seq_k)
        sum_grad_attn = np.sum(attn * grad_attn, axis=-1, keepdims=True)
        grad_scores = attn * (grad_attn - sum_grad_attn)

        # Apply mask gradient (masked positions have zero gradient)
        if mask is not None:
            grad_scores = np.where(mask == 1, 0.0, grad_scores)

        # Scale factor gradient
        grad_scores = grad_scores / math.sqrt(self.d_k)

        # Gradient w.r.t. query: d/dQ (QK^T) = grad_scores @ K
        # grad_scores: (batch, heads, seq_q, seq_k)
        # key: (batch, heads, seq_k, d_k)
        # grad_query: (batch, heads, seq_q, d_k)
        grad_query = np.matmul(grad_scores, key)

        # Gradient w.r.t. key: d/dK (QK^T) = grad_scores^T @ Q
        # grad_key: (batch, heads, seq_k, d_k)
        grad_key = np.matmul(grad_scores.transpose(0, 1, 3, 2), query)

        return grad_query, grad_key, grad_value


# =============================================================================
# Multi-Head Attention
# =============================================================================


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

    Args:
        d_model: Model dimension (input/output dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability

    Attributes:
        W_Q: Query projection weights, shape (d_model, d_model)
        W_K: Key projection weights, shape (d_model, d_model)
        W_V: Value projection weights, shape (d_model, d_model)
        W_O: Output projection weights, shape (d_model, d_model)
        b_Q, b_K, b_V, b_O: Bias terms

    Notes:
        - d_model must be divisible by num_heads
        - Each head operates on d_k = d_model // num_heads dimensions
        - Parameters are NOT shared between heads (each head has its own slice)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.dropout = dropout

        # Initialize weights using Xavier initialization
        std = math.sqrt(2.0 / (d_model + d_model))

        # Each head has independent parameters (stored as slices of these matrices)
        self.W_Q = np.random.randn(d_model, d_model) * std
        self.W_K = np.random.randn(d_model, d_model) * std
        self.W_V = np.random.randn(d_model, d_model) * std
        self.W_O = np.random.randn(d_model, d_model) * std

        # Bias terms
        self.b_Q = np.zeros(d_model)
        self.b_K = np.zeros(d_model)
        self.b_V = np.zeros(d_model)
        self.b_O = np.zeros(d_model)

        # Attention layer
        self.attention = ScaledDotProductAttention(self.d_k, dropout)

        # Cache for backward pass
        self.cache = None

        # Gradients
        self.grad_W_Q = None
        self.grad_W_K = None
        self.grad_W_V = None
        self.grad_W_O = None
        self.grad_b_Q = None
        self.grad_b_K = None
        self.grad_b_V = None
        self.grad_b_O = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Reshaped tensor, shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # Reshape to (batch, seq_len, num_heads, d_k) then transpose
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back to original dimension.

        Args:
            x: Input tensor, shape (batch, num_heads, seq_len, d_k)

        Returns:
            Reshaped tensor, shape (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose to (batch, seq_len, num_heads, d_k) then reshape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor, shape (batch, seq_q, d_model)
            key: Key tensor, shape (batch, seq_k, d_model)
            value: Value tensor, shape (batch, seq_v, d_model)
                   Note: seq_k == seq_v for self-attention
            mask: Optional mask, shape (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)

        Returns:
            output: Attention output, shape (batch, seq_q, d_model)
        """
        query = _ensure_array(query)
        key = _ensure_array(key)
        value = _ensure_array(value)

        batch_size = query.shape[0]

        # Linear projections: (batch, seq, d_model) @ (d_model, d_model) + (d_model,)
        Q = np.matmul(query, self.W_Q) + self.b_Q
        K = np.matmul(key, self.W_K) + self.b_K
        V = np.matmul(value, self.W_V) + self.b_V

        # Split into multiple heads
        Q = self._split_heads(Q)  # (batch, num_heads, seq_q, d_k)
        K = self._split_heads(K)  # (batch, num_heads, seq_k, d_k)
        V = self._split_heads(V)  # (batch, num_heads, seq_v, d_k)

        # Apply scaled dot-product attention
        attn_output = self.attention.forward(Q, K, V, mask)  # (batch, num_heads, seq_q, d_k)

        # Combine heads
        attn_output = self._combine_heads(attn_output)  # (batch, seq_q, d_model)

        # Final linear projection
        output = np.matmul(attn_output, self.W_O) + self.b_O

        # Cache for backward pass
        self.cache = (query, key, value, Q, K, V, attn_output, mask)

        return output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for multi-head attention.

        Args:
            grad_output: Gradient from next layer, shape (batch, seq_q, d_model)

        Returns:
            grad_query: Gradient w.r.t. query, shape (batch, seq_q, d_model)
            grad_key: Gradient w.r.t. key, shape (batch, seq_k, d_model)
            grad_value: Gradient w.r.t. value, shape (batch, seq_v, d_model)
        """
        if self.cache is None:
            raise RuntimeError("Must call forward before backward")

        query, key, value, Q, K, V, attn_output, mask = self.cache

        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Gradient through output projection
        # grad_W_O = attn_output^T @ grad_output (summed over batch)
        self.grad_W_O = np.matmul(
            attn_output.reshape(-1, self.d_model).T,
            grad_output.reshape(-1, self.d_model)
        )
        self.grad_b_O = np.sum(grad_output, axis=(0, 1))

        # Gradient w.r.t. attn_output
        grad_attn_output = np.matmul(grad_output, self.W_O.T)

        # Split heads for backward through attention
        grad_attn_split = self._split_heads(grad_attn_output)

        # Backward through scaled dot-product attention
        grad_Q, grad_K, grad_V = self.attention.backward(grad_attn_split)

        # Combine heads
        grad_Q = self._combine_heads(grad_Q)
        grad_K = self._combine_heads(grad_K)
        grad_V = self._combine_heads(grad_V)

        # Gradient through input projections
        self.grad_W_Q = np.matmul(
            query.reshape(-1, self.d_model).T,
            grad_Q.reshape(-1, self.d_model)
        )
        self.grad_W_K = np.matmul(
            key.reshape(-1, self.d_model).T,
            grad_K.reshape(-1, self.d_model)
        )
        self.grad_W_V = np.matmul(
            value.reshape(-1, self.d_model).T,
            grad_V.reshape(-1, self.d_model)
        )

        self.grad_b_Q = np.sum(grad_Q, axis=(0, 1))
        self.grad_b_K = np.sum(grad_K, axis=(0, 1))
        self.grad_b_V = np.sum(grad_V, axis=(0, 1))

        # Gradient w.r.t. inputs
        grad_query = np.matmul(grad_Q, self.W_Q.T)
        grad_key = np.matmul(grad_K, self.W_K.T)
        grad_value = np.matmul(grad_V, self.W_V.T)

        return grad_query, grad_key, grad_value

    def parameters(self) -> list:
        """Return list of all parameters."""
        return [self.W_Q, self.W_K, self.W_V, self.W_O,
                self.b_Q, self.b_K, self.b_V, self.b_O]

    def gradients(self) -> list:
        """Return list of all gradients."""
        return [self.grad_W_Q, self.grad_W_K, self.grad_W_V, self.grad_W_O,
                self.grad_b_Q, self.grad_b_K, self.grad_b_V, self.grad_b_O]

    def zero_grad(self):
        """Reset all gradients to None."""
        self.grad_W_Q = None
        self.grad_W_K = None
        self.grad_W_V = None
        self.grad_W_O = None
        self.grad_b_Q = None
        self.grad_b_K = None
        self.grad_b_V = None
        self.grad_b_O = None


# =============================================================================
# Sinusoidal Positional Encoding
# =============================================================================


class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding as described in "Attention Is All You Need".

    Uses sine and cosine functions of different frequencies to encode position:

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This encoding has useful properties:
    - Unique for each position (no learned parameters)
    - Can extrapolate to longer sequences than seen during training
    - Relative positions can be computed as linear functions

    Args:
        d_model: Model dimension (must match embedding dimension)
        max_len: Maximum sequence length to precompute
        dropout: Dropout probability (applied after adding positional encoding)

    Attributes:
        pe: Precomputed positional encodings, shape (1, max_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = dropout

        # Precompute positional encodings
        self.pe = self._create_positional_encoding(max_len, d_model)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """
        Create the positional encoding matrix.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding matrix, shape (1, max_len, d_model)
        """
        # Create position indices: (max_len, 1)
        position = np.arange(max_len)[:, np.newaxis]

        # Create dimension indices for division: (1, d_model//2)
        # Use exp(log) for numerical stability: 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = np.exp(np.arange(0, d_model, 2) * np.log(10000.0) / d_model)

        # Create encoding matrix
        pe = np.zeros((max_len, d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(position / div_term)  # Even indices
        pe[:, 1::2] = np.cos(position / div_term)  # Odd indices

        # Add batch dimension: (1, max_len, d_model)
        pe = pe[np.newaxis, :]

        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)

        Returns:
            Output with positional encoding added, shape (batch, seq_len, d_model)
        """
        x = _ensure_array(x)
        seq_len = x.shape[1]

        # Add positional encoding (broadcast over batch dimension)
        return x + self.pe[:, :seq_len, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass - positional encoding has no learnable parameters.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient passed through unchanged (identity)
        """
        return grad_output

    def get_encoding(self, seq_len: int) -> np.ndarray:
        """
        Get positional encoding for a specific sequence length.

        Args:
            seq_len: Desired sequence length

        Returns:
            Positional encoding, shape (1, seq_len, d_model)
        """
        return self.pe[:, :seq_len, :].copy()

    def is_unique(self, seq_len: int, tolerance: float = 1e-6) -> bool:
        """
        Check if all positional encodings are unique within a sequence length.

        Args:
            seq_len: Sequence length to check
            tolerance: Tolerance for comparing positions

        Returns:
            True if all positions have unique encodings
        """
        encodings = self.pe[0, :seq_len, :]  # (seq_len, d_model)

        # Compare each position with all others
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                distance = np.linalg.norm(encodings[i] - encodings[j])
                if distance < tolerance:
                    return False
        return True


# =============================================================================
# Transformer Encoder Layer
# =============================================================================


class TransformerEncoderLayer:
    """
    Single Transformer Encoder Layer.

    Consists of:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-Forward Network (two linear layers with ReLU)
    4. Add & Norm

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (typically 4 * d_model)
        dropout: Dropout probability

    Notes:
        - Layer normalization is applied before sub-layers (Pre-LN) for stability
        - Uses simplified layer normalization (without learnable parameters)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network weights
        std_ff = math.sqrt(2.0 / (d_model + self.d_ff))
        self.W1 = np.random.randn(d_model, self.d_ff) * std_ff
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * std_ff
        self.b2 = np.zeros(d_model)

        # Layer normalization parameters (simplified - no learnable params)
        self.eps = 1e-6

        # Cache for backward pass
        self.cache = None

        # Gradients
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Simplified layer normalization (no learnable parameters).

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)

    def _layer_norm_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Backward pass for layer normalization.

        Args:
            grad_output: Gradient from next layer
            x: Original input before normalization

        Returns:
            Gradient w.r.t. input
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)

        x_norm = (x - mean) / std
        n = x.shape[-1]

        # Gradient of layer norm
        dx_norm = grad_output
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

        grad_input = dx_norm / std + dvar * 2.0 * (x - mean) / n + dmean / n
        return grad_input

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU."""
        return grad_output * (x > 0).astype(np.float64)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass for encoder layer.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        x = _ensure_array(x)

        # 1. Multi-Head Self-Attention with residual
        attn_output = self.self_attn.forward(x, x, x, mask)
        x = self._layer_norm(x + attn_output)
        attn_residual = x

        # 2. Feed-Forward Network with residual
        ff_hidden = self._relu(np.matmul(x, self.W1) + self.b1)
        ff_output = np.matmul(ff_hidden, self.W2) + self.b2
        x = self._layer_norm(x + ff_output)

        # Cache for backward
        self.cache = (x, attn_residual, ff_hidden, ff_output, mask)

        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for encoder layer.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient w.r.t. input
        """
        if self.cache is None:
            raise RuntimeError("Must call forward before backward")

        x, attn_residual, ff_hidden, ff_output, mask = self.cache

        # Backward through second layer norm and FFN
        grad = self._layer_norm_backward(grad_output, attn_residual + ff_output)
        grad_ff = grad
        grad_attn_res = grad

        # FFN gradients
        self.grad_W2 = np.matmul(ff_hidden.reshape(-1, self.d_ff).T, grad_ff.reshape(-1, self.d_model))
        self.grad_b2 = np.sum(grad_ff, axis=(0, 1))

        grad_ff_hidden = np.matmul(grad_ff, self.W2.T)
        grad_ff_hidden = self._relu_backward(grad_ff_hidden, ff_hidden)

        self.grad_W1 = np.matmul(attn_residual.reshape(-1, self.d_model).T, grad_ff_hidden.reshape(-1, self.d_ff))
        self.grad_b1 = np.sum(grad_ff_hidden, axis=(0, 1))

        grad_attn_res += np.matmul(grad_ff_hidden, self.W1.T)

        # Backward through first layer norm and attention
        grad = self._layer_norm_backward(grad_attn_res, attn_residual)

        # Backward through attention
        grad_q, grad_k, grad_v = self.self_attn.backward(grad)
        grad_input = grad + grad_q  # For self-attention, all are same

        return grad_input

    def parameters(self) -> list:
        """Return list of all parameters."""
        return (
            self.self_attn.parameters() +
            [self.W1, self.b1, self.W2, self.b2]
        )

    def gradients(self) -> list:
        """Return list of all gradients."""
        return (
            self.self_attn.gradients() +
            [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]
        )

    def zero_grad(self):
        """Reset all gradients."""
        self.self_attn.zero_grad()
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None


# =============================================================================
# Utility Functions
# =============================================================================


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (autoregressive) mask for decoder self-attention.

    Prevents positions from attending to subsequent positions.

    Args:
        seq_len: Sequence length

    Returns:
        Mask tensor, shape (1, 1, seq_len, seq_len)
        Positions to mask are 1, positions to attend are 0
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask[np.newaxis, np.newaxis, :, :]


def create_padding_mask(seq: np.ndarray, pad_idx: int = 0) -> np.ndarray:
    """
    Create a padding mask from sequence indices.

    Args:
        seq: Sequence indices, shape (batch, seq_len)
        pad_idx: Padding token index

    Returns:
        Mask tensor, shape (batch, 1, 1, seq_len)
        Padding positions are 1, non-padding are 0
    """
    mask = (seq == pad_idx).astype(np.float64)
    return mask[:, np.newaxis, np.newaxis, :]


def count_parameters_attention(model: Union[MultiHeadAttention, TransformerEncoderLayer]) -> int:
    """
    Count total trainable parameters in attention model.

    Args:
        model: Attention model

    Returns:
        Total number of parameters
    """
    total = 0
    for param in model.parameters():
        total += param.size
    return total


# =============================================================================
# Registry
# =============================================================================

ATTENTION_COMPONENTS = {
    "scaled_dot_product": ScaledDotProductAttention,
    "multi_head": MultiHeadAttention,
    "sinusoidal_pe": SinusoidalPositionalEncoding,
    "encoder_layer": TransformerEncoderLayer,
}
