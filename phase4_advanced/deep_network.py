"""
Deep Network Experiments for Gradient Stability.

This module provides:
    - Deep MLP with configurable depth for gradient flow testing
    - Deep ResNet with skip connections for gradient preservation
    - Deep LSTM with layer normalization for gradient stability
    - Gradient flow experiments and benchmarks

Theory:
    Why deep networks have gradient problems:

    MLP Chain Rule:
        dL/dW1 = dL/dL * dL/dL-1 * ... * d2/d1 * d1/dW1
        If each gradient is < 1, product vanishes exponentially
        If each gradient is > 1, product explodes exponentially

    ResNet Solution:
        y = F(x) + x
        dy/dx = dF/dx + 1
        The "+1" ensures gradient >= 1, preventing vanishing

    LSTM Solution:
        c_t = f_t * c_{t-1} + i_t * g_t
        Gradient can flow through c_{t-1} multiplied by f_t
        Learnable forget gate can be close to 1

References:
    - Deep Residual Learning for Image Recognition (He et al., 2015)
    - Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
    - Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
"""

from typing import List, Optional, Tuple, Dict, Any, Callable, Union
from dataclasses import dataclass
import numpy as np

ArrayLike = Union[np.ndarray, List, float]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array with float64 dtype."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)
    return x


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """ReLU gradient."""
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(x)


# =============================================================================
# Deep MLP (No Skip Connections) - Demonstrates Vanishing Gradients
# =============================================================================


class DeepMLP:
    """
    Deep MLP without skip connections.

    This network demonstrates the vanishing gradient problem.
    As depth increases, gradients in early layers become exponentially small.

    Architecture:
        Input -> [Linear -> ReLU] x num_layers -> Output

    Gradient Flow:
        Each ReLU has max gradient 1, so gradients compound multiplicatively.
        For L layers: gradient_scale ~ (0.5)^L (assuming half of ReLUs are active)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ):
        """
        Initialize DeepMLP.

        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
            num_layers: Number of hidden layers
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize weights with He initialization
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Input layer
        std = np.sqrt(2.0 / input_size)
        self.weights.append(np.random.randn(input_size, hidden_size) * std)
        self.biases.append(np.zeros(hidden_size))

        # Hidden layers
        std = np.sqrt(2.0 / hidden_size)
        for _ in range(num_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * std)
            self.biases.append(np.zeros(hidden_size))

        # Output layer
        std = np.sqrt(2.0 / hidden_size)
        self.weights.append(np.random.randn(hidden_size, output_size) * std)
        self.biases.append(np.zeros(output_size))

        # Cache for backward
        self._activations: List[np.ndarray] = []
        self._pre_activations: List[np.ndarray] = []

        # Gradients
        self.grad_weights: List[Optional[np.ndarray]] = [None] * len(self.weights)
        self.grad_biases: List[Optional[np.ndarray]] = [None] * len(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, input_size)

        Returns:
            Output tensor, shape (batch, output_size)
        """
        x = _ensure_array(x)
        self._activations = [x]
        self._pre_activations = []

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            self._pre_activations.append(z)
            x = relu(z)
            self._activations.append(x)

        # Output layer (no activation)
        z = x @ self.weights[-1] + self.biases[-1]
        self._pre_activations.append(z)
        self._activations.append(z)

        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient from loss, shape (batch, output_size)

        Returns:
            Gradient w.r.t. input
        """
        grad_output = _ensure_array(grad_output)

        # Output layer
        x = self._activations[-2]
        self.grad_weights[-1] = x.T @ grad_output
        self.grad_biases[-1] = np.sum(grad_output, axis=0)
        grad = grad_output

        # Hidden layers (reverse order)
        for i in range(len(self.weights) - 2, -1, -1):
            # Gradient through weights
            grad = grad @ self.weights[i + 1].T

            # Gradient through ReLU
            grad = grad * relu_grad(self._pre_activations[i])

            # Compute weight gradients
            x = self._activations[i]
            self.grad_weights[i] = x.T @ grad
            self.grad_biases[i] = np.sum(grad, axis=0)

        # Final gradient through first weight layer
        grad = grad @ self.weights[0].T

        return grad

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w)
            params.append(b)
        return params

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return all gradients."""
        grads = []
        for gw, gb in zip(self.grad_weights, self.grad_biases):
            grads.append(gw)
            grads.append(gb)
        return grads

    def get_layer_gradient_norms(self) -> List[float]:
        """Get gradient norms for each layer."""
        norms = []
        for gw, gb in zip(self.grad_weights, self.grad_biases):
            if gw is not None:
                norm = np.sqrt(np.sum(gw**2) + np.sum(gb**2))
                norms.append(float(norm))
        return norms


# =============================================================================
# Deep ResNet (With Skip Connections) - Solves Vanishing Gradients
# =============================================================================


class ResidualBlock:
    """
    Basic residual block with two conv-like linear layers.

    Architecture:
        x -> Linear -> ReLU -> Linear -> ReLU + x (skip)
    """

    def __init__(self, hidden_size: int):
        """Initialize residual block."""
        self.hidden_size = hidden_size

        # Two linear layers
        std = np.sqrt(2.0 / hidden_size)
        self.W1 = np.random.randn(hidden_size, hidden_size) * std
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * std
        self.b2 = np.zeros(hidden_size)

        # Cache
        self._cache: Dict[str, np.ndarray] = {}
        self._grads: Dict[str, Optional[np.ndarray]] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with skip connection."""
        self._cache["x"] = x

        # First linear + ReLU
        z1 = x @ self.W1 + self.b1
        self._cache["z1"] = z1
        a1 = relu(z1)

        # Second linear
        z2 = a1 @ self.W2 + self.b2
        self._cache["z2"] = z2
        self._cache["a1"] = a1

        # Skip connection + ReLU
        out = relu(z2 + x)  # F(x) + x
        self._cache["out"] = out

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with skip connection."""
        x = self._cache["x"]
        z1 = self._cache["z1"]
        a1 = self._cache["a1"]
        z2 = self._cache["z2"]

        # Gradient through output ReLU
        grad = grad_output * relu_grad(z2 + x)

        # Gradient for W2, b2
        self._grads["W2"] = a1.T @ grad
        self._grads["b2"] = np.sum(grad, axis=0)

        # Gradient through W2
        grad = grad @ self.W2.T

        # Gradient through first ReLU
        grad = grad * relu_grad(z1)

        # Gradient for W1, b1
        self._grads["W1"] = x.T @ grad
        self._grads["b1"] = np.sum(grad, axis=0)

        # Gradient through W1
        grad = grad @ self.W1.T

        # Add skip connection gradient (+1)
        grad = grad + grad_output * relu_grad(z2 + x)

        return grad

    def parameters(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def gradients(self) -> List[Optional[np.ndarray]]:
        return [
            self._grads.get("W1"),
            self._grads.get("b1"),
            self._grads.get("W2"),
            self._grads.get("b2"),
        ]


class DeepResNet:
    """
    Deep ResNet with skip connections.

    Skip connections ensure gradient flow:
        dL/dx = dL/d(output) * (dF/dx + 1)

    The "+1" term ensures gradients don't vanish even in very deep networks.

    Architecture:
        Input -> Linear -> [ResBlock] x num_blocks -> Linear -> Output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_blocks: int,
    ):
        """
        Initialize DeepResNet.

        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
            num_blocks: Number of residual blocks
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_blocks = num_blocks

        # Input projection
        std = np.sqrt(2.0 / input_size)
        self.W_in = np.random.randn(input_size, hidden_size) * std
        self.b_in = np.zeros(hidden_size)

        # Residual blocks
        self.blocks = [ResidualBlock(hidden_size) for _ in range(num_blocks)]

        # Output projection
        std = np.sqrt(2.0 / hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * std
        self.b_out = np.zeros(output_size)

        # Cache
        self._cache: Dict[str, np.ndarray] = {}
        self._grads: Dict[str, Optional[np.ndarray]] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        x = _ensure_array(x)
        self._cache["x_in"] = x

        # Input projection
        h = x @ self.W_in + self.b_in
        h = relu(h)
        self._cache["h_in"] = h

        # Residual blocks
        for block in self.blocks:
            h = block.forward(h)

        self._cache["h_final"] = h

        # Output projection
        out = h @ self.W_out + self.b_out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        grad_output = _ensure_array(grad_output)
        h = self._cache["h_final"]

        # Output layer gradients
        self._grads["W_out"] = h.T @ grad_output
        self._grads["b_out"] = np.sum(grad_output, axis=0)
        grad = grad_output @ self.W_out.T

        # Residual blocks (reverse order)
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        # Input projection gradients
        x = self._cache["x_in"]
        h_in = self._cache["h_in"]
        grad = grad * relu_grad(h_in)
        self._grads["W_in"] = x.T @ grad
        self._grads["b_in"] = np.sum(grad, axis=0)

        return grad @ self.W_in.T

    def parameters(self) -> List[np.ndarray]:
        """Return all parameters."""
        params = [self.W_in, self.b_in]
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend([self.W_out, self.b_out])
        return params

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return all gradients."""
        grads = [self._grads.get("W_in"), self._grads.get("b_in")]
        for block in self.blocks:
            grads.extend(block.gradients())
        grads.extend([self._grads.get("W_out"), self._grads.get("b_out")])
        return grads

    def get_layer_gradient_norms(self) -> List[float]:
        """Get gradient norms for each layer."""
        norms = []

        # Input layer
        if self._grads.get("W_in") is not None:
            norm = np.sqrt(np.sum(self._grads["W_in"] ** 2) + np.sum(self._grads["b_in"] ** 2))
            norms.append(float(norm))

        # Residual blocks
        for block in self.blocks:
            grads = block.gradients()
            total = 0.0
            for g in grads:
                if g is not None:
                    total += np.sum(g**2)
            norms.append(float(np.sqrt(total)))

        # Output layer
        if self._grads.get("W_out") is not None:
            norm = np.sqrt(np.sum(self._grads["W_out"] ** 2) + np.sum(self._grads["b_out"] ** 2))
            norms.append(float(norm))

        return norms


# =============================================================================
# Deep LSTM (With Layer Normalization)
# =============================================================================


class LSTMLayerNorm:
    """Layer Normalization for LSTM."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = np.ones(hidden_size)
        self.beta = np.zeros(hidden_size)
        self._cache = {}
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self._cache["x"] = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self._cache["mean"] = mean
        self._cache["var"] = var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        self._cache["x_norm"] = x_norm
        return self.gamma * x_norm + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        mean = self._cache["mean"]
        var = self._cache["var"]
        n = self.hidden_size

        self.grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(x.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(x.ndim - 1)))

        dx_norm = grad_output * self.gamma
        std = np.sqrt(var + self.eps)
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / std, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        grad_input = dx_norm / std + dvar * 2 * (x - mean) / n + dmean / n
        return grad_input


class DeepLSTMCell:
    """
    LSTM Cell with Layer Normalization for deep networks.

    Layer normalization helps stabilize gradients in deep LSTMs.

    Gates:
        f_t = sigmoid(W_f @ [x, h] + b_f)  (forget gate)
        i_t = sigmoid(W_i @ [x, h] + b_i)  (input gate)
        g_t = tanh(W_g @ [x, h] + b_g)     (cell candidate)
        o_t = sigmoid(W_o @ [x, h] + b_o)  (output gate)

    Cell state:
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, use_layer_norm: bool = True):
        """Initialize LSTM cell."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm

        # Combined weight matrix for all gates
        std = np.sqrt(2.0 / (input_size + hidden_size))
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * std
        self.b = np.zeros(4 * hidden_size)

        # Layer normalization (optional)
        if use_layer_norm:
            self.ln_i = LSTMLayerNorm(hidden_size)
            self.ln_f = LSTMLayerNorm(hidden_size)
            self.ln_g = LSTMLayerNorm(hidden_size)
            self.ln_o = LSTMLayerNorm(hidden_size)
        else:
            self.ln_i = self.ln_f = self.ln_g = self.ln_o = None

        # Cache
        self._cache: Dict[str, np.ndarray] = {}
        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass for single timestep."""
        x = _ensure_array(x)
        h_prev = _ensure_array(h_prev)
        c_prev = _ensure_array(c_prev)

        # Concatenate input and hidden
        combined = np.concatenate([x, h_prev], axis=-1)
        self._cache["combined"] = combined

        # Compute all gates at once
        gates = combined @ self.W + self.b
        self._cache["gates"] = gates

        # Split gates
        i, f, g, o = np.split(gates, 4, axis=-1)

        # Apply layer norm if enabled
        if self.use_layer_norm:
            i = self.ln_i.forward(i)
            f = self.ln_f.forward(f)
            g = self.ln_g.forward(g)
            o = self.ln_o.forward(o)

        self._cache["i_raw"] = i
        self._cache["f_raw"] = f
        self._cache["g_raw"] = g
        self._cache["o_raw"] = o

        # Apply activations
        i_t = sigmoid(i)
        f_t = sigmoid(f)
        g_t = tanh(g)
        o_t = sigmoid(o)

        self._cache["i_t"] = i_t
        self._cache["f_t"] = f_t
        self._cache["g_t"] = g_t
        self._cache["o_t"] = o_t
        self._cache["c_prev"] = c_prev

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        self._cache["c_t"] = c_t

        # Update hidden state
        h_t = o_t * tanh(c_t)
        self._cache["tanh_c_t"] = tanh(c_t)

        return h_t, c_t

    def backward(
        self, grad_h: np.ndarray, grad_c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass."""
        i_t = self._cache["i_t"]
        f_t = self._cache["f_t"]
        g_t = self._cache["g_t"]
        o_t = self._cache["o_t"]
        c_t = self._cache["c_t"]
        c_prev = self._cache["c_prev"]
        tanh_c_t = self._cache["tanh_c_t"]
        combined = self._cache["combined"]

        # Gradient through h_t = o_t * tanh(c_t)
        grad_o = grad_h * tanh_c_t
        grad_c = grad_c + grad_h * o_t * (1 - tanh_c_t**2)

        # Gradient through c_t = f_t * c_prev + i_t * g_t
        grad_f = grad_c * c_prev
        grad_i = grad_c * g_t
        grad_g = grad_c * i_t
        grad_c_prev = grad_c * f_t

        # Gradient through activations
        grad_i_raw = grad_i * i_t * (1 - i_t)
        grad_f_raw = grad_f * f_t * (1 - f_t)
        grad_g_raw = grad_g * (1 - g_t**2)
        grad_o_raw = grad_o * o_t * (1 - o_t)

        # Back through layer norm if enabled
        if self.use_layer_norm:
            grad_i_raw = self.ln_i.backward(grad_i_raw)
            grad_f_raw = self.ln_f.backward(grad_f_raw)
            grad_g_raw = self.ln_g.backward(grad_g_raw)
            grad_o_raw = self.ln_o.backward(grad_o_raw)

        # Concatenate gate gradients
        grad_gates = np.concatenate([grad_i_raw, grad_f_raw, grad_g_raw, grad_o_raw], axis=-1)

        # Weight gradients
        self.grad_W = combined.T @ grad_gates
        self.grad_b = np.sum(grad_gates, axis=0)

        # Gradient w.r.t. combined input
        grad_combined = grad_gates @ self.W.T

        # Split into x and h_prev gradients
        grad_x = grad_combined[:, : self.input_size]
        grad_h_prev = grad_combined[:, self.input_size :]

        return grad_x, grad_h_prev, grad_c_prev


class DeepLSTM:
    """
    Multi-layer Deep LSTM with Layer Normalization.

    Stacks multiple LSTM layers, each with layer normalization
    to ensure stable gradient flow in very deep networks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        use_layer_norm: bool = True,
    ):
        """
        Initialize Deep LSTM.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            use_layer_norm: Whether to use layer normalization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create LSTM cells for each layer
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(DeepLSTMCell(in_size, hidden_size, use_layer_norm))

        # Cache for backward
        self._states_cache: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
        c0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through all layers and timesteps.

        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h0: Initial hidden states, shape (num_layers, batch, hidden_size)
            c0: Initial cell states, shape (num_layers, batch, hidden_size)

        Returns:
            Tuple of (output, h_n, c_n)
        """
        x = _ensure_array(x)
        batch_size, seq_len, _ = x.shape

        # Initialize states
        if h0 is None:
            h0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((self.num_layers, batch_size, self.hidden_size))

        # Transpose for layer-major processing
        h = h0.copy()  # (num_layers, batch, hidden)
        c = c0.copy()

        # Cache for backward
        self._states_cache = []

        # Process sequence
        outputs = []
        for t in range(seq_len):
            layer_states = []
            x_t = x[:, t, :]

            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(x_t, h[layer_idx], c[layer_idx])
                layer_states.append((h[layer_idx].copy(), c[layer_idx].copy()))
                x_t = h[layer_idx]  # Input to next layer

            outputs.append(h[-1])  # Output from last layer
            self._states_cache.append(layer_states)

        # Stack outputs
        output = np.stack(outputs, axis=1)  # (batch, seq_len, hidden)

        return output, h, c

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through time (BPTT).

        Args:
            grad_output: Gradient w.r.t. output, shape (batch, seq_len, hidden_size)

        Returns:
            Gradient w.r.t. input
        """
        batch_size, seq_len, _ = grad_output.shape

        # Initialize gradients
        grad_h = np.zeros((self.num_layers, batch_size, self.hidden_size))
        grad_c = np.zeros((self.num_layers, batch_size, self.hidden_size))
        grad_x = np.zeros((batch_size, seq_len, self.input_size))

        # Backward through time (reverse order)
        for t in range(seq_len - 1, -1, -1):
            # Gradient from output
            grad_h[-1] = grad_h[-1] + grad_output[:, t, :]

            # Backward through layers (reverse order)
            for layer_idx in range(self.num_layers - 1, -1, -1):
                cell = self.cells[layer_idx]
                grad_x_t, grad_h_prev, grad_c_prev = cell.backward(grad_h[layer_idx], grad_c[layer_idx])

                grad_c[layer_idx] = grad_c_prev

                if layer_idx > 0:
                    grad_h[layer_idx - 1] = grad_h[layer_idx - 1] + grad_x_t
                else:
                    grad_x[:, t, :] = grad_x_t

                grad_h[layer_idx] = grad_h_prev

        return grad_x

    def get_layer_gradient_norms(self) -> List[float]:
        """Get gradient norms for each layer."""
        norms = []
        for cell in self.cells:
            if cell.grad_W is not None:
                norm = np.sqrt(np.sum(cell.grad_W**2) + np.sum(cell.grad_b**2))
                norms.append(float(norm))
        return norms


# =============================================================================
# Gradient Flow Experiment Functions
# =============================================================================


def run_gradient_flow_experiment(
    model_class: str,
    depth: int,
    input_size: int = 64,
    hidden_size: int = 128,
    batch_size: int = 32,
    seq_len: int = 10,
) -> Dict[str, Any]:
    """
    Run gradient flow experiment on a deep network.

    Args:
        model_class: 'mlp', 'resnet', or 'lstm'
        depth: Number of layers/blocks
        input_size: Input dimension
        hidden_size: Hidden dimension
        batch_size: Batch size
        seq_len: Sequence length (for LSTM)

    Returns:
        Dictionary with experiment results
    """
    np.random.seed(42)

    if model_class == "mlp":
        model = DeepMLP(input_size, hidden_size, hidden_size, depth)
        x = np.random.randn(batch_size, input_size)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)
        layer_norms = model.get_layer_gradient_norms()

    elif model_class == "resnet":
        model = DeepResNet(input_size, hidden_size, hidden_size, depth)
        x = np.random.randn(batch_size, input_size)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)
        layer_norms = model.get_layer_gradient_norms()

    elif model_class == "lstm":
        model = DeepLSTM(input_size, hidden_size, depth, use_layer_norm=True)
        x = np.random.randn(batch_size, seq_len, input_size)
        output, _, _ = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)
        layer_norms = model.get_layer_gradient_norms()

    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Compute flow statistics
    if layer_norms:
        first_norm = layer_norms[0] if layer_norms[0] > 0 else 1e-10
        flow_ratios = [n / first_norm for n in layer_norms]
        min_ratio = min(flow_ratios)
        max_ratio = max(flow_ratios)
    else:
        flow_ratios = []
        min_ratio = 0
        max_ratio = 0

    return {
        "model_class": model_class,
        "depth": depth,
        "layer_norms": layer_norms,
        "flow_ratios": flow_ratios,
        "min_flow_ratio": min_ratio,
        "max_flow_ratio": max_ratio,
        "is_vanishing": min_ratio < 0.01,
        "is_exploding": max_ratio > 100,
    }


def compare_gradient_flow(
    depths: List[int] = [10, 20, 50, 100],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare gradient flow across different architectures and depths.

    Args:
        depths: List of depths to test

    Returns:
        Dictionary with results for each architecture
    """
    results = {"mlp": [], "resnet": [], "lstm": []}

    for depth in depths:
        for model_class in ["mlp", "resnet", "lstm"]:
            result = run_gradient_flow_experiment(model_class, depth)
            results[model_class].append(result)

    return results
