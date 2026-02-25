"""
Unit tests for activation functions.

Tests verify:
1. Forward pass output dimensions
2. Analytical gradient matches numerical gradient (< 1e-6 error)
3. Specific gradient values at key points
4. PyTorch comparison for consistency
"""

import math

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase1_basics.activations import (
    gelu,
    gelu_grad,
    leaky_relu,
    leaky_relu_grad,
    numerical_gradient,
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    swish,
    swish_grad,
    tanh,
    tanh_grad,
    _erf,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_input():
    """Generate random input for testing."""
    np.random.seed(42)
    return np.random.randn(10, 5) * 2


@pytest.fixture
def small_input():
    """Small input for simple tests."""
    return np.array([0.0, 1.0, -1.0, 2.0, -2.0])


# =============================================================================
# Sigmoid Tests
# =============================================================================


class TestSigmoid:
    """Test suite for sigmoid activation."""

    def test_sigmoid_forward_shape(self, random_input):
        """Test sigmoid forward pass preserves shape."""
        result = sigmoid(random_input)
        assert result.shape == random_input.shape

    def test_sigmoid_output_range(self):
        """Test sigmoid output is approximately in (0, 1) for typical inputs."""
        # Use moderate values where sigmoid doesn't saturate
        x = np.array([-5, -2, -1, 0, 1, 2, 5])
        result = sigmoid(x)
        assert np.all(result > 0) and np.all(result < 1)

    def test_sigmoid_at_zero(self):
        """Test sigmoid(0) = 0.5."""
        assert np.isclose(sigmoid(0), 0.5)

    def test_sigmoid_gradient_at_zero(self):
        """Test sigmoid gradient at x=0 equals 0.25.

        This is a key success criterion from Octie task.
        """
        grad = sigmoid_grad(0)
        assert np.isclose(grad, 0.25), f"Expected 0.25, got {grad}"

    def test_sigmoid_gradient_numerical(self, random_input):
        """Test sigmoid gradient matches numerical gradient."""
        analytical = sigmoid_grad(random_input)
        numerical = numerical_gradient(lambda x: np.sum(sigmoid(x)), random_input.copy())
        assert np.allclose(analytical, numerical, atol=1e-6), (
            f"Max error: {np.max(np.abs(analytical - numerical))}"
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_sigmoid_pytorch_comparison(self, random_input):
        """Test sigmoid matches PyTorch implementation."""
        numpy_out = sigmoid(random_input)
        torch_out = torch.sigmoid(torch.tensor(random_input, dtype=torch.float64)).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-6)


# =============================================================================
# Tanh Tests
# =============================================================================


class TestTanh:
    """Test suite for tanh activation."""

    def test_tanh_forward_shape(self, random_input):
        """Test tanh forward pass preserves shape."""
        result = tanh(random_input)
        assert result.shape == random_input.shape

    def test_tanh_output_range(self):
        """Test tanh output is approximately in (-1, 1) for typical inputs."""
        # Use moderate values where tanh doesn't saturate completely
        x = np.array([-5, -2, -1, 0, 1, 2, 5])
        result = tanh(x)
        assert np.all(result > -1) and np.all(result < 1)

    def test_tanh_at_zero(self):
        """Test tanh(0) = 0."""
        assert np.isclose(tanh(0), 0.0)

    def test_tanh_gradient_at_zero(self):
        """Test tanh gradient at x=0 equals 1."""
        grad = tanh_grad(0)
        assert np.isclose(grad, 1.0), f"Expected 1.0, got {grad}"

    def test_tanh_gradient_numerical(self, random_input):
        """Test tanh gradient matches numerical gradient."""
        analytical = tanh_grad(random_input)
        numerical = numerical_gradient(lambda x: np.sum(tanh(x)), random_input.copy())
        assert np.allclose(analytical, numerical, atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_tanh_pytorch_comparison(self, random_input):
        """Test tanh matches PyTorch implementation."""
        numpy_out = tanh(random_input)
        torch_out = torch.tanh(torch.tensor(random_input, dtype=torch.float64)).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-6)


# =============================================================================
# ReLU Tests
# =============================================================================


class TestReLU:
    """Test suite for ReLU activation."""

    def test_relu_forward_shape(self, random_input):
        """Test ReLU forward pass preserves shape."""
        result = relu(random_input)
        assert result.shape == random_input.shape

    def test_relu_positive_passthrough(self):
        """Test ReLU passes through positive values unchanged."""
        x = np.array([0.5, 1.0, 2.0, 10.0])
        result = relu(x)
        assert np.allclose(result, x)

    def test_relu_negative_zero(self):
        """Test ReLU outputs zero for negative inputs."""
        x = np.array([-0.5, -1.0, -2.0, -10.0])
        result = relu(x)
        assert np.allclose(result, 0.0)

    def test_relu_gradient_positive(self):
        """Test ReLU gradient is 1 for x > 0.

        This is a key success criterion from Octie task.
        """
        x = np.array([0.5, 1.0, 2.0, 10.0])
        grad = relu_grad(x)
        assert np.allclose(grad, 1.0), f"Expected all 1.0, got {grad}"

    def test_relu_gradient_negative(self):
        """Test ReLU gradient is 0 for x < 0.

        This is a key success criterion from Octie task.
        """
        x = np.array([-0.5, -1.0, -2.0, -10.0])
        grad = relu_grad(x)
        assert np.allclose(grad, 0.0), f"Expected all 0.0, got {grad}"

    def test_relu_gradient_at_zero(self):
        """Test ReLU gradient at x=0 (convention: 0)."""
        grad = relu_grad(0)
        # Convention: gradient at 0 is 0 (subgradient)
        assert grad == 0.0

    def test_relu_gradient_numerical(self, random_input):
        """Test ReLU gradient matches numerical gradient."""
        analytical = relu_grad(random_input)
        numerical = numerical_gradient(lambda x: np.sum(relu(x)), random_input.copy())
        # Use slightly larger tolerance for ReLU due to discontinuity
        assert np.allclose(analytical, numerical, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_relu_pytorch_comparison(self, random_input):
        """Test ReLU matches PyTorch implementation."""
        numpy_out = relu(random_input)
        torch_out = torch.relu(torch.tensor(random_input, dtype=torch.float64)).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-6)


# =============================================================================
# Leaky ReLU Tests
# =============================================================================


class TestLeakyReLU:
    """Test suite for Leaky ReLU activation."""

    def test_leaky_relu_forward_shape(self, random_input):
        """Test Leaky ReLU forward pass preserves shape."""
        result = leaky_relu(random_input)
        assert result.shape == random_input.shape

    def test_leaky_relu_positive_passthrough(self):
        """Test Leaky ReLU passes through positive values unchanged."""
        x = np.array([0.5, 1.0, 2.0])
        result = leaky_relu(x)
        assert np.allclose(result, x)

    def test_leaky_relu_negative_scaled(self):
        """Test Leaky ReLU scales negative values by alpha."""
        alpha = 0.1
        x = np.array([-0.5, -1.0, -2.0])
        result = leaky_relu(x, alpha=alpha)
        expected = x * alpha
        assert np.allclose(result, expected)

    def test_leaky_relu_gradient_positive(self):
        """Test Leaky ReLU gradient is 1 for x > 0."""
        x = np.array([0.5, 1.0, 2.0])
        grad = leaky_relu_grad(x)
        assert np.allclose(grad, 1.0)

    def test_leaky_relu_gradient_negative(self):
        """Test Leaky ReLU gradient is alpha for x < 0."""
        alpha = 0.1
        x = np.array([-0.5, -1.0, -2.0])
        grad = leaky_relu_grad(x, alpha=alpha)
        assert np.allclose(grad, alpha)

    def test_leaky_relu_gradient_numerical(self, random_input):
        """Test Leaky ReLU gradient matches numerical gradient."""
        alpha = 0.01
        analytical = leaky_relu_grad(random_input, alpha=alpha)
        numerical = numerical_gradient(
            lambda x: np.sum(leaky_relu(x, alpha=alpha)), random_input.copy()
        )
        assert np.allclose(analytical, numerical, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_leaky_relu_pytorch_comparison(self, random_input):
        """Test Leaky ReLU matches PyTorch implementation."""
        alpha = 0.01
        numpy_out = leaky_relu(random_input, alpha=alpha)
        torch_out = torch.nn.functional.leaky_relu(
            torch.tensor(random_input, dtype=torch.float64), negative_slope=alpha
        ).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-6)


# =============================================================================
# GELU Tests
# =============================================================================


class TestGELU:
    """Test suite for GELU activation."""

    def test_gelu_forward_shape(self, random_input):
        """Test GELU forward pass preserves shape."""
        result = gelu(random_input)
        assert result.shape == random_input.shape

    def test_gelu_at_zero(self):
        """Test GELU(0) = 0 (since x * Φ(x) = 0 * 0.5 = 0).

        This is a key success criterion from Octie task.
        """
        result = gelu(0)
        assert np.isclose(result, 0.0, atol=1e-6), f"Expected 0.0, got {result}"

    def test_gelu_at_zero_exact(self):
        """Test exact GELU at x=0."""
        result = gelu(0, approximate=False)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_gelu_gradient_at_zero(self):
        """Test GELU gradient at x=0 equals 0.5."""
        grad = gelu_grad(0)
        # At x=0: Φ(0) = 0.5, φ(0) = 1/√(2π), so grad = 0.5 + 0 = 0.5
        assert np.isclose(grad, 0.5, atol=1e-6), f"Expected 0.5, got {grad}"

    def test_gelu_approximation_accuracy(self):
        """Test GELU approximation is reasonably close to exact."""
        x = np.linspace(-3, 3, 100)
        approx = gelu(x, approximate=True)
        exact = gelu(x, approximate=False)
        # Approximation should be within 10% relative error (tanh approximation)
        # Note: approximation error is higher near zero but still small absolute error
        abs_error = np.abs(approx - exact)
        # Check absolute error is small (max ~0.0002)
        assert np.max(abs_error) < 0.001, f"Max absolute error: {np.max(abs_error)}"

    def test_gelu_gradient_numerical_approx(self, random_input):
        """Test approximate GELU gradient matches numerical gradient."""
        analytical = gelu_grad(random_input, approximate=True)
        numerical = numerical_gradient(
            lambda x: np.sum(gelu(x, approximate=True)), random_input.copy()
        )
        assert np.allclose(analytical, numerical, atol=1e-5)

    def test_gelu_gradient_numerical_exact(self, random_input):
        """Test exact GELU gradient matches numerical gradient."""
        analytical = gelu_grad(random_input, approximate=False)
        numerical = numerical_gradient(
            lambda x: np.sum(gelu(x, approximate=False)), random_input.copy()
        )
        assert np.allclose(analytical, numerical, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_gelu_pytorch_comparison(self, random_input):
        """Test GELU matches PyTorch implementation."""
        numpy_out = gelu(random_input, approximate=True)
        torch_out = torch.nn.functional.gelu(
            torch.tensor(random_input, dtype=torch.float64), approximate="tanh"
        ).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-5)


# =============================================================================
# Swish Tests
# =============================================================================


class TestSwish:
    """Test suite for Swish activation."""

    def test_swish_forward_shape(self, random_input):
        """Test Swish forward pass preserves shape."""
        result = swish(random_input)
        assert result.shape == random_input.shape

    def test_swish_at_zero(self):
        """Test Swish(0) = 0."""
        assert np.isclose(swish(0), 0.0)

    def test_swish_gradient_at_zero(self):
        """Test Swish gradient at x=0 equals 0.5."""
        grad = swish_grad(0)
        assert np.isclose(grad, 0.5, atol=1e-6), f"Expected 0.5, got {grad}"

    def test_swish_self_gated(self):
        """Test Swish is self-gated (x * sigmoid(x))."""
        x = np.array([0.5, 1.0, 2.0])
        result = swish(x)
        expected = x * sigmoid(x)
        assert np.allclose(result, expected)

    def test_swish_gradient_numerical(self, random_input):
        """Test Swish gradient matches numerical gradient."""
        analytical = swish_grad(random_input)
        numerical = numerical_gradient(lambda x: np.sum(swish(x)), random_input.copy())
        assert np.allclose(analytical, numerical, atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_swish_pytorch_comparison(self, random_input):
        """Test Swish matches PyTorch SiLU implementation."""
        numpy_out = swish(random_input)
        torch_out = torch.nn.functional.silu(
            torch.tensor(random_input, dtype=torch.float64)
        ).numpy()
        assert np.allclose(numpy_out, torch_out, atol=1e-6)


# =============================================================================
# Error Function Tests
# =============================================================================


class TestErf:
    """Test suite for error function approximation."""

    def test_erf_at_zero(self):
        """Test erf(0) = 0."""
        assert np.isclose(_erf(np.array([0.0]))[0], 0.0, atol=1e-7)

    def test_erf_symmetry(self):
        """Test erf(-x) = -erf(x)."""
        x = np.array([0.5, 1.0, 2.0])
        assert np.allclose(_erf(-x), -_erf(x), atol=1e-7)

    def test_erf_asymptotic_positive(self):
        """Test erf(x) → 1 as x → +∞."""
        x = np.array([5.0, 10.0])
        result = _erf(x)
        assert np.all(result > 0.999)

    def test_erf_asymptotic_negative(self):
        """Test erf(x) → -1 as x → -∞."""
        x = np.array([-5.0, -10.0])
        result = _erf(x)
        assert np.all(result < -0.999)

    def test_erf_known_values(self):
        """Test erf at known values."""
        # erf(1) ≈ 0.8427
        assert np.isclose(_erf(np.array([1.0]))[0], 0.8427007929497, atol=1e-7)


# =============================================================================
# Numerical Gradient Tests
# =============================================================================


class TestNumericalGradient:
    """Test suite for numerical gradient function."""

    def test_numerical_gradient_linear(self):
        """Test numerical gradient of linear function."""
        x = np.array([2.0])
        grad = numerical_gradient(lambda x: 3 * x[0], x)
        assert np.isclose(grad[0], 3.0, atol=1e-5)

    def test_numerical_gradient_quadratic(self):
        """Test numerical gradient of quadratic function."""
        x = np.array([2.0])
        grad = numerical_gradient(lambda x: x[0] ** 2, x)
        # d/dx(x²) = 2x = 4 at x=2
        assert np.isclose(grad[0], 4.0, atol=1e-5)

    def test_numerical_gradient_multidim(self):
        """Test numerical gradient with multidimensional input."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = numerical_gradient(lambda x: np.sum(x**2), x)
        # d/dx(sum(x²)) = 2x
        expected = 2 * x
        assert np.allclose(grad, expected, atol=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple activation functions."""

    def test_all_activations_same_shape(self, random_input):
        """Test all activations preserve input shape."""
        activations = [sigmoid, tanh, relu, leaky_relu, gelu, swish]
        for act in activations:
            result = act(random_input)
            assert result.shape == random_input.shape, (
                f"{act.__name__} changed shape from {random_input.shape} to {result.shape}"
            )

    def test_all_gradients_same_shape(self, random_input):
        """Test all gradients preserve input shape."""
        gradients = [sigmoid_grad, tanh_grad, relu_grad, leaky_relu_grad, gelu_grad, swish_grad]
        for grad_fn in gradients:
            result = grad_fn(random_input)
            assert result.shape == random_input.shape, (
                f"{grad_fn.__name__} changed shape from {random_input.shape} to {result.shape}"
            )

    def test_gradient_flow_in_transformer_context(self):
        """Test ReLU and GELU gradient flow as in Transformer.

        This is a key success criterion from Octie task.
        Simulates gradient flowing through a feedforward block.
        """
        np.random.seed(42)
        batch_size = 4
        hidden_dim = 64
        ff_dim = 256

        # Simulate feedforward block: Linear -> Activation -> Linear
        W1 = np.random.randn(hidden_dim, ff_dim) * 0.02
        W2 = np.random.randn(ff_dim, hidden_dim) * 0.02
        x = np.random.randn(batch_size, hidden_dim)

        # Forward with ReLU
        h1 = x @ W1
        h1_relu = relu(h1)
        out_relu = h1_relu @ W2

        # Forward with GELU
        h1_gelu = gelu(h1)
        out_gelu = h1_gelu @ W2

        # Backward simulation (gradient = ones)
        grad_out = np.ones_like(out_relu)

        # Gradient through second linear
        grad_h1_relu = grad_out @ W2.T
        grad_h1_gelu = grad_out @ W2.T

        # Gradient through activations
        grad_relu = relu_grad(h1) * grad_h1_relu
        grad_gelu = gelu_grad(h1) * grad_h1_gelu

        # Gradients should flow (not all zeros)
        assert not np.allclose(grad_relu, 0), "ReLU gradient is all zeros"
        assert not np.allclose(grad_gelu, 0), "GELU gradient is all zeros"

        # Gradient dimensions should match
        assert grad_relu.shape == (batch_size, ff_dim)
        assert grad_gelu.shape == (batch_size, ff_dim)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_sigmoid_extreme_values(self):
        """Test sigmoid handles extreme values without overflow."""
        x = np.array([-1000, -100, 100, 1000])
        result = sigmoid(x)
        assert np.all(np.isfinite(result))
        # For extreme values, sigmoid approaches 0 or 1 but may not equal exactly
        assert result[0] < 1e-100  # -1000 → nearly 0
        assert result[-1] >= 1 - 1e-100  # 1000 → nearly 1 (may be exactly 1)

    def test_relu_extreme_values(self):
        """Test ReLU handles extreme values."""
        x = np.array([-1e10, -1000, 1000, 1e10])
        result = relu(x)
        assert np.all(np.isfinite(result))

    def test_gelu_extreme_values(self):
        """Test GELU handles extreme values."""
        x = np.array([-100, -10, 10, 100])
        result = gelu(x)
        assert np.all(np.isfinite(result))

    def test_very_small_inputs(self):
        """Test activations handle very small inputs."""
        x = np.array([1e-10, -1e-10, 1e-15, -1e-15])
        for act in [sigmoid, tanh, relu, gelu, swish]:
            result = act(x)
            assert np.all(np.isfinite(result)), f"{act.__name__} failed on small inputs"
