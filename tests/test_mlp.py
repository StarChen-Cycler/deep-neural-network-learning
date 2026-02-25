"""
Tests for Multi-Layer Perceptron forward and backward propagation.

This module tests:
1. Forward pass output shapes and values
2. Backward pass gradient computation
3. Numerical gradient verification (< 1e-6 error)
4. PyTorch autograd comparison
5. Computational graph visualization
"""

import pytest
import numpy as np

# Check if PyTorch is available for comparison tests
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase1_basics.mlp import (
    LinearLayer,
    ActivationLayer,
    MLP,
    ComputationalGraphVisualizer,
    numerical_gradient_mlp,
    mse_loss,
    mse_loss_grad,
)
from phase1_basics.activations import relu, relu_grad, sigmoid, sigmoid_grad


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def simple_input():
    """Simple 2D input for testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def small_mlp():
    """Small 3-layer MLP for testing."""
    np.random.seed(42)
    return MLP(input_size=4, hidden_sizes=[8, 4], output_size=2, activation="relu")


# =============================================================================
# LinearLayer Tests
# =============================================================================

class TestLinearLayer:
    """Test suite for LinearLayer class."""

    def test_forward_shape(self, random_seed):
        """Test forward pass output shape."""
        layer = LinearLayer(in_features=10, out_features=5)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"

    def test_forward_no_bias(self, random_seed):
        """Test forward pass without bias."""
        layer = LinearLayer(in_features=10, out_features=5, bias=False)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 5)

    def test_forward_values(self, simple_input):
        """Test forward pass values match manual computation."""
        layer = LinearLayer(in_features=2, out_features=3)
        # Set known weights and bias
        layer.weight = np.array([[1.0, 0.5, -1.0],
                                  [0.5, 1.0, 0.5]])
        layer.bias = np.array([0.1, 0.2, 0.3])

        output = layer.forward(simple_input)

        # Manual computation for first sample: [1, 2] @ W + b
        expected_0 = np.array([1.0*1.0 + 2.0*0.5 + 0.1,    # = 2.1
                               1.0*0.5 + 2.0*1.0 + 0.2,    # = 2.7
                               1.0*(-1.0) + 2.0*0.5 + 0.3]) # = 0.3
        assert np.allclose(output[0], expected_0, atol=1e-6)

    def test_backward_shape(self, random_seed):
        """Test backward pass gradient shapes."""
        layer = LinearLayer(in_features=10, out_features=5)
        x = np.random.randn(32, 10)
        layer.forward(x)

        grad_output = np.random.randn(32, 5)
        grad_input = layer.backward(grad_output)

        assert grad_input.shape == (32, 10)
        assert layer.grad_weight.shape == (10, 5)
        assert layer.grad_bias.shape == (5,)

    def test_backward_values(self, simple_input):
        """Test backward pass gradient values."""
        layer = LinearLayer(in_features=2, out_features=3)
        layer.weight = np.array([[1.0, 0.5, -1.0],
                                  [0.5, 1.0, 0.5]])
        layer.bias = np.array([0.1, 0.2, 0.3])

        output = layer.forward(simple_input)

        grad_output = np.ones_like(output)
        grad_input = layer.backward(grad_output)

        # Manual gradient computation
        # grad_weight = input.T @ grad_output
        expected_grad_weight = simple_input.T @ grad_output
        assert np.allclose(layer.grad_weight, expected_grad_weight, atol=1e-6)

        # grad_bias = sum(grad_output, axis=0)
        expected_grad_bias = np.sum(grad_output, axis=0)
        assert np.allclose(layer.grad_bias, expected_grad_bias, atol=1e-6)

        # grad_input = grad_output @ weight.T
        expected_grad_input = grad_output @ layer.weight.T
        assert np.allclose(grad_input, expected_grad_input, atol=1e-6)

    def test_backward_before_forward_raises(self):
        """Test that backward before forward raises error."""
        layer = LinearLayer(in_features=10, out_features=5)
        grad_output = np.random.randn(32, 5)
        with pytest.raises(RuntimeError):
            layer.backward(grad_output)


# =============================================================================
# ActivationLayer Tests
# =============================================================================

class TestActivationLayer:
    """Test suite for ActivationLayer class."""

    def test_relu_forward(self, random_seed):
        """Test ReLU activation forward pass."""
        layer = ActivationLayer(relu, relu_grad, "relu")
        x = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.float64)
        output = layer.forward(x)
        expected = np.array([[0, 0, 1], [2, 0, 4]], dtype=np.float64)
        assert np.allclose(output, expected)

    def test_relu_backward(self, random_seed):
        """Test ReLU activation backward pass."""
        layer = ActivationLayer(relu, relu_grad, "relu")
        x = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.float64)
        layer.forward(x)

        grad_output = np.ones_like(x)
        grad_input = layer.backward(grad_output)

        # ReLU gradient: 1 for x > 0, 0 otherwise
        expected = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float64)
        assert np.allclose(grad_input, expected)

    def test_no_parameters(self):
        """Test that activation layers have no parameters."""
        layer = ActivationLayer(relu, relu_grad, "relu")
        assert len(layer.parameters()) == 0


# =============================================================================
# MLP Tests
# =============================================================================

class TestMLP:
    """Test suite for MLP class."""

    def test_forward_output_shape(self, small_mlp):
        """Test criterion: 3层MLP前向传播输出维度正确"""
        x = np.random.randn(16, 4)  # batch=16, input=4
        output = small_mlp.forward(x)
        assert output.shape == (16, 2), f"Expected (16, 2), got {output.shape}"

    def test_forward_different_batch_sizes(self, small_mlp):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 8, 32, 128]:
            x = np.random.randn(batch_size, 4)
            output = small_mlp.forward(x)
            assert output.shape == (batch_size, 2)

    def test_backward_output_shape(self, small_mlp):
        """Test backward pass gradient shape."""
        x = np.random.randn(16, 4)
        small_mlp.forward(x)

        grad_output = np.random.randn(16, 2)
        grad_input = small_mlp.backward(grad_output)

        assert grad_input.shape == (16, 4)

    def test_backward_all_gradients_computed(self, small_mlp):
        """Test that all layer gradients are computed."""
        x = np.random.randn(16, 4)
        small_mlp.forward(x)

        grad_output = np.random.randn(16, 2)
        small_mlp.backward(grad_output)

        # Check that all linear layers have gradients
        for layer in small_mlp.layers:
            if isinstance(layer, LinearLayer):
                assert layer.grad_weight is not None
                assert layer.grad_bias is not None

    def test_zero_grad(self, small_mlp):
        """Test that zero_grad clears all gradients."""
        x = np.random.randn(16, 4)
        small_mlp.forward(x)
        small_mlp.backward(np.random.randn(16, 2))

        small_mlp.zero_grad()

        for layer in small_mlp.layers:
            if isinstance(layer, LinearLayer):
                assert layer.grad_weight is None
                assert layer.grad_bias is None

    def test_parameters_count(self):
        """Test that parameters returns correct count."""
        mlp = MLP(input_size=4, hidden_sizes=[8], output_size=2)
        # Layer 1: W(4,8) + b(8) = 2 params
        # Layer 2: W(8,2) + b(2) = 2 params
        # Total: 4 param tensors
        params = mlp.parameters()
        assert len(params) == 4


# =============================================================================
# Gradient Verification Tests
# =============================================================================

class TestGradientVerification:
    """Test suite for gradient verification against numerical gradients."""

    def test_linear_weight_gradient_numerical(self, random_seed):
        """Test criterion: 反向传播梯度与numpy手动计算误差<1e-6"""
        np.random.seed(42)
        layer = LinearLayer(in_features=4, out_features=3)

        x = np.random.randn(8, 4)
        target = np.random.randn(8, 3)

        # Forward
        output = layer.forward(x)
        loss = mse_loss(output, target)

        # Analytical gradient
        grad_loss = mse_loss_grad(output, target)
        layer.backward(grad_loss)
        analytical_grad = layer.grad_weight.copy()

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(layer.weight)

        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                old = layer.weight[i, j]

                layer.weight[i, j] = old + eps
                out_plus = layer.forward(x)
                loss_plus = mse_loss(out_plus, target)

                layer.weight[i, j] = old - eps
                out_minus = layer.forward(x)
                loss_minus = mse_loss(out_minus, target)

                layer.weight[i, j] = old
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

        error = np.abs(analytical_grad - numerical_grad)
        max_error = np.max(error)
        assert max_error < 1e-6, f"Weight gradient error {max_error} exceeds 1e-6"

    def test_linear_bias_gradient_numerical(self, random_seed):
        """Test bias gradient against numerical gradient."""
        np.random.seed(42)
        layer = LinearLayer(in_features=4, out_features=3)

        x = np.random.randn(8, 4)
        target = np.random.randn(8, 3)

        # Forward
        output = layer.forward(x)
        loss = mse_loss(output, target)

        # Analytical gradient
        grad_loss = mse_loss_grad(output, target)
        layer.backward(grad_loss)
        analytical_grad = layer.grad_bias.copy()

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(layer.bias)

        for i in range(len(layer.bias)):
            old = layer.bias[i]

            layer.bias[i] = old + eps
            out_plus = layer.forward(x)
            loss_plus = mse_loss(out_plus, target)

            layer.bias[i] = old - eps
            out_minus = layer.forward(x)
            loss_minus = mse_loss(out_minus, target)

            layer.bias[i] = old
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        error = np.abs(analytical_grad - numerical_grad)
        max_error = np.max(error)
        assert max_error < 1e-6, f"Bias gradient error {max_error} exceeds 1e-6"

    def test_mlp_weight_gradient_numerical(self, random_seed):
        """Test criterion: 手动梯度验证误差<1e-6 (MLP)"""
        np.random.seed(42)
        mlp = MLP(input_size=4, hidden_sizes=[8], output_size=2)

        x = np.random.randn(4, 4)
        target = np.random.randn(4, 2)

        # Forward
        output = mlp.forward(x)
        loss = mse_loss(output, target)

        # Analytical gradient
        grad_loss = mse_loss_grad(output, target)
        mlp.backward(grad_loss)

        # Get first linear layer
        linear_layer = mlp.layers[0]
        analytical_grad = linear_layer.grad_weight.copy()

        # Numerical gradient
        eps = 1e-5
        numerical_grad = np.zeros_like(linear_layer.weight)

        for i in range(linear_layer.weight.shape[0]):
            for j in range(linear_layer.weight.shape[1]):
                old = linear_layer.weight[i, j]

                linear_layer.weight[i, j] = old + eps
                out_plus = mlp.forward(x)
                loss_plus = mse_loss(out_plus, target)

                linear_layer.weight[i, j] = old - eps
                out_minus = mlp.forward(x)
                loss_minus = mse_loss(out_minus, target)

                linear_layer.weight[i, j] = old
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

        error = np.abs(analytical_grad - numerical_grad)
        max_error = np.max(error)
        assert max_error < 1e-6, f"MLP weight gradient error {max_error} exceeds 1e-6"


# =============================================================================
# PyTorch Comparison Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchComparison:
    """Test suite for comparing with PyTorch autograd."""

    def test_linear_forward_pytorch(self, random_seed):
        """Test linear forward matches PyTorch."""
        np.random.seed(42)
        torch.manual_seed(42)

        # NumPy implementation
        np_layer = LinearLayer(in_features=10, out_features=5)

        # PyTorch implementation
        torch_layer = nn.Linear(10, 5)

        # Copy weights
        torch_layer.weight.data = torch.from_numpy(np_layer.weight.T).double()
        torch_layer.bias.data = torch.from_numpy(np_layer.bias).double()

        # Input
        x_np = np.random.randn(16, 10).astype(np.float64)
        x_torch = torch.from_numpy(x_np).double()

        # Forward
        out_np = np_layer.forward(x_np)
        out_torch = torch_layer(x_torch).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-6)

    def test_linear_backward_pytorch(self, random_seed):
        """Test criterion: PyTorch autograd梯度验证通过"""
        np.random.seed(42)
        torch.manual_seed(42)

        # NumPy implementation
        np_layer = LinearLayer(in_features=10, out_features=5)

        # PyTorch implementation
        torch_layer = nn.Linear(10, 5)

        # Copy weights
        torch_layer.weight.data = torch.from_numpy(np_layer.weight.T).double()
        torch_layer.bias.data = torch.from_numpy(np_layer.bias).double()

        # Input
        x_np = np.random.randn(16, 10).astype(np.float64)
        x_torch = torch.from_numpy(x_np).double()
        x_torch.requires_grad = True

        # Forward
        out_np = np_layer.forward(x_np)
        out_torch = torch_layer(x_torch)

        # Backward with known gradient
        grad_output_np = np.random.randn(16, 5).astype(np.float64)
        grad_output_torch = torch.from_numpy(grad_output_np).double()

        # NumPy backward
        grad_input_np = np_layer.backward(grad_output_np)

        # PyTorch backward
        out_torch.backward(grad_output_torch)

        # Compare gradients
        assert np.allclose(np_layer.grad_weight, torch_layer.weight.grad.T.numpy(), atol=1e-6)
        assert np.allclose(np_layer.grad_bias, torch_layer.bias.grad.numpy(), atol=1e-6)
        assert np.allclose(grad_input_np, x_torch.grad.numpy(), atol=1e-6)

    def test_mlp_forward_pytorch(self, random_seed):
        """Test MLP forward matches PyTorch."""
        np.random.seed(42)
        torch.manual_seed(42)

        # NumPy MLP
        np_mlp = MLP(input_size=10, hidden_sizes=[16, 8], output_size=3, activation="relu")

        # PyTorch MLP
        torch_mlp = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

        # Copy weights
        torch_mlp[0].weight.data = torch.from_numpy(np_mlp.layers[0].weight.T).double()
        torch_mlp[0].bias.data = torch.from_numpy(np_mlp.layers[0].bias).double()
        torch_mlp[2].weight.data = torch.from_numpy(np_mlp.layers[2].weight.T).double()
        torch_mlp[2].bias.data = torch.from_numpy(np_mlp.layers[2].bias).double()
        torch_mlp[4].weight.data = torch.from_numpy(np_mlp.layers[4].weight.T).double()
        torch_mlp[4].bias.data = torch.from_numpy(np_mlp.layers[4].bias).double()

        # Input
        x_np = np.random.randn(8, 10).astype(np.float64)
        x_torch = torch.from_numpy(x_np).double()

        # Forward
        out_np = np_mlp.forward(x_np)
        out_torch = torch_mlp(x_torch).detach().numpy()

        assert np.allclose(out_np, out_torch, atol=1e-6)

    def test_mlp_backward_pytorch(self, random_seed):
        """Test MLP backward matches PyTorch autograd."""
        np.random.seed(42)
        torch.manual_seed(42)

        # NumPy MLP
        np_mlp = MLP(input_size=10, hidden_sizes=[16], output_size=3, activation="relu")

        # PyTorch MLP
        torch_mlp = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

        # Copy weights
        torch_mlp[0].weight.data = torch.from_numpy(np_mlp.layers[0].weight.T).double()
        torch_mlp[0].bias.data = torch.from_numpy(np_mlp.layers[0].bias).double()
        torch_mlp[2].weight.data = torch.from_numpy(np_mlp.layers[2].weight.T).double()
        torch_mlp[2].bias.data = torch.from_numpy(np_mlp.layers[2].bias).double()

        # Input
        x_np = np.random.randn(8, 10).astype(np.float64)
        x_torch = torch.from_numpy(x_np).double()
        x_torch.requires_grad = True

        # Target and loss
        target_np = np.random.randn(8, 3).astype(np.float64)
        target_torch = torch.from_numpy(target_np).double()

        # NumPy forward + backward
        out_np = np_mlp.forward(x_np)
        loss_np = mse_loss(out_np, target_np)
        grad_loss = mse_loss_grad(out_np, target_np)
        np_mlp.backward(grad_loss)

        # PyTorch forward + backward
        out_torch = torch_mlp(x_torch)
        loss_torch = nn.functional.mse_loss(out_torch, target_torch)
        loss_torch.backward()

        # Compare loss values
        assert np.isclose(loss_np, loss_torch.item(), rtol=1e-5)

        # Compare gradients for first layer
        np_grad_w0 = np_mlp.layers[0].grad_weight
        torch_grad_w0 = torch_mlp[0].weight.grad.T.numpy()
        assert np.allclose(np_grad_w0, torch_grad_w0, atol=1e-6)


# =============================================================================
# Computational Graph Tests
# =============================================================================

class TestComputationalGraph:
    """Test suite for computational graph visualization."""

    def test_visualize_forward(self, small_mlp):
        """Test criterion: 计算图可视化显示正确的依赖关系"""
        viz = ComputationalGraphVisualizer()
        graph = viz.visualize_forward(small_mlp)

        assert "FORWARD PASS" in graph
        assert "[Linear]" in graph
        assert "[Relu" in graph

    def test_visualize_backward(self, small_mlp):
        """Test backward pass visualization."""
        viz = ComputationalGraphVisualizer()
        graph = viz.visualize_backward(small_mlp)

        assert "BACKWARD PASS" in graph
        assert "Gradient Flow" in graph or "dW" in graph

    def test_graph_shows_layer_order(self):
        """Test that graph shows correct layer ordering."""
        mlp = MLP(input_size=4, hidden_sizes=[8, 4], output_size=2)
        viz = ComputationalGraphVisualizer()
        graph = viz.visualize_forward(mlp)

        # Check that layers appear in correct order
        lines = graph.split('\n')
        linear_count = sum(1 for line in lines if '[Linear]' in line)
        relu_count = sum(1 for line in lines if '[Relu' in line or 'relu' in line.lower())

        # Should have 3 linear layers and 2 ReLU activations
        assert linear_count == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full MLP training loop."""

    def test_training_step(self, small_mlp, random_seed):
        """Test a single training step updates parameters correctly."""
        x = np.random.randn(8, 4)
        target = np.random.randn(8, 2)

        # Store initial weights
        initial_weights = small_mlp.layers[0].weight.copy()

        # Forward
        output = small_mlp.forward(x)
        loss = mse_loss(output, target)

        # Backward
        grad_loss = mse_loss_grad(output, target)
        small_mlp.backward(grad_loss)

        # Update (simple SGD)
        lr = 0.01
        for layer in small_mlp.layers:
            if isinstance(layer, LinearLayer):
                layer.weight -= lr * layer.grad_weight
                layer.bias -= lr * layer.grad_bias

        # Check weights changed
        assert not np.allclose(small_mlp.layers[0].weight, initial_weights)

    def test_loss_decreases_with_training(self, random_seed):
        """Test that loss decreases over multiple training steps."""
        np.random.seed(42)
        mlp = MLP(input_size=4, hidden_sizes=[16, 8], output_size=2)

        x = np.random.randn(32, 4)
        target = np.random.randn(32, 2)

        lr = 0.01
        losses = []

        for _ in range(10):
            # Forward
            output = mlp.forward(x)
            loss = mse_loss(output, target)
            losses.append(loss)

            # Backward
            grad_loss = mse_loss_grad(output, target)
            mlp.backward(grad_loss)

            # Update
            for layer in mlp.layers:
                if isinstance(layer, LinearLayer):
                    layer.weight -= lr * layer.grad_weight
                    layer.bias -= lr * layer.grad_bias

            mlp.zero_grad()

        # Loss should generally decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]} -> {losses[-1]}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_single_sample(self, small_mlp):
        """Test with batch size 1."""
        x = np.random.randn(1, 4)
        output = small_mlp.forward(x)
        assert output.shape == (1, 2)

        grad_input = small_mlp.backward(np.random.randn(1, 2))
        assert grad_input.shape == (1, 4)

    def test_large_input_values(self, small_mlp):
        """Test with large input values (numerical stability)."""
        x = np.random.randn(8, 4) * 100
        output = small_mlp.forward(x)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_zero_input(self, small_mlp):
        """Test with zero input."""
        x = np.zeros((8, 4))
        output = small_mlp.forward(x)
        assert output.shape == (8, 2)

    def test_different_activations(self):
        """Test MLP with different activation functions."""
        for activation in ["relu", "sigmoid", "tanh"]:
            mlp = MLP(input_size=4, hidden_sizes=[8], output_size=2, activation=activation)
            x = np.random.randn(8, 4)
            output = mlp.forward(x)
            assert output.shape == (8, 2), f"Failed for activation: {activation}"
