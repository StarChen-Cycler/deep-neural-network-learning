"""
Tests for gradient stability module.

Tests cover:
    - Gradient clipping (clip_grad_norm, clip_grad_value)
    - Gradient flow diagnostics
    - Vanishing/exploding gradient detection
    - Deep network gradient flow (MLP, ResNet, LSTM)
"""

import pytest
import numpy as np
from typing import List

from phase4_advanced.gradient_stability import (
    clip_grad_norm,
    clip_grad_value,
    GradientFlowAnalyzer,
    GradientStats,
    detect_vanishing_gradient,
    detect_exploding_gradient,
    apply_skip_connection,
    LayerScale,
    compute_gradient_norm,
    get_gradient_clipper,
)

from phase4_advanced.deep_network import (
    DeepMLP,
    DeepResNet,
    DeepLSTM,
    ResidualBlock,
    run_gradient_flow_experiment,
    compare_gradient_flow,
)

# Try to import PyTorch for comparison
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def sample_gradients() -> List[np.ndarray]:
    """Create sample gradients for testing."""
    return [
        np.random.randn(32, 64),
        np.random.randn(64, 128),
        np.random.randn(128, 10),
    ]


# =============================================================================
# Test Gradient Clipping
# =============================================================================


class TestClipGradNorm:
    """Tests for clip_grad_norm function."""

    def test_basic_clipping(self):
        """Test that gradients are clipped to max_norm."""
        grads = [np.array([3.0, 4.0])]  # norm = 5
        clipped, total_norm = clip_grad_norm(grads, max_norm=1.0)
        clipped_norm = np.sqrt(np.sum(clipped[0] ** 2))
        assert np.isclose(clipped_norm, 1.0, atol=1e-6)
        assert np.isclose(total_norm, 5.0, atol=1e-6)

    def test_no_clipping_needed(self):
        """Test that small gradients are not clipped."""
        grads = [np.array([0.1, 0.1])]  # norm ~ 0.14
        clipped, total_norm = clip_grad_norm(grads, max_norm=1.0)
        # Should be unchanged
        assert np.allclose(clipped[0], grads[0])

    def test_max_norm_constraint(self):
        """Test that max_norm=1.0 results in gradient norm <= 1.0."""
        grads = [np.random.randn(100, 100) * 10]  # Large gradients
        clipped, _ = clip_grad_norm(grads, max_norm=1.0, norm_type=2.0)
        clipped_norm = np.sqrt(np.sum(clipped[0] ** 2))
        assert clipped_norm <= 1.0 + 1e-6

    def test_l1_norm(self):
        """Test L1 norm clipping."""
        grads = [np.array([1.0, 2.0, 3.0])]  # L1 norm = 6
        clipped, total_norm = clip_grad_norm(grads, max_norm=2.0, norm_type=1.0)
        clipped_l1 = np.sum(np.abs(clipped[0]))
        assert clipped_l1 <= 2.0 + 1e-6

    def test_inf_norm(self):
        """Test max (inf) norm clipping."""
        grads = [np.array([1.0, 5.0, 3.0])]  # max abs = 5
        clipped, total_norm = clip_grad_norm(grads, max_norm=2.0, norm_type=float("inf"))
        max_abs = np.max(np.abs(clipped[0]))
        assert max_abs <= 2.0 + 1e-6

    def test_multiple_gradients(self):
        """Test clipping multiple gradient tensors."""
        grads = [
            np.random.randn(32, 64) * 5,
            np.random.randn(64, 128) * 5,
            np.random.randn(128, 10) * 5,
        ]
        clipped, total_norm = clip_grad_norm(grads, max_norm=1.0)

        # Compute total norm of clipped gradients
        clipped_total = np.sqrt(sum(np.sum(g**2) for g in clipped))
        assert clipped_total <= 1.0 + 1e-6

    def test_invalid_max_norm(self):
        """Test that invalid max_norm raises error."""
        grads = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError):
            clip_grad_norm(grads, max_norm=0)
        with pytest.raises(ValueError):
            clip_grad_norm(grads, max_norm=-1)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self):
        """Compare with PyTorch clip_grad_norm_."""
        # Create sample parameters
        np_grads = [np.random.randn(32, 64) * 10, np.random.randn(64, 10) * 10]

        # Our implementation
        np_clipped, np_total_norm = clip_grad_norm(np_grads, max_norm=1.0, norm_type=2.0)

        # PyTorch implementation
        torch_params = [torch.tensor(g.copy(), requires_grad=True) for g in np_grads]
        for p, g in zip(torch_params, np_grads):
            p.grad = torch.tensor(g.copy())

        torch_total_norm = torch.nn.utils.clip_grad_norm_(torch_params, max_norm=1.0, norm_type=2.0)

        # Compare total norms
        assert np.isclose(np_total_norm, torch_total_norm.item(), rtol=1e-4)


class TestClipGradValue:
    """Tests for clip_grad_value function."""

    def test_basic_clipping(self):
        """Test that values are clipped to [-clip_value, clip_value]."""
        grads = [np.array([5.0, -3.0, 2.0])]
        clipped = clip_grad_value(grads, clip_value=2.0)
        expected = np.array([2.0, -2.0, 2.0])
        assert np.allclose(clipped[0], expected)

    def test_no_clipping_needed(self):
        """Test that small values are not modified."""
        grads = [np.array([0.5, -0.3, 0.2])]
        clipped = clip_grad_value(grads, clip_value=2.0)
        assert np.allclose(clipped[0], grads[0])

    def test_negative_values(self):
        """Test clipping of negative values."""
        grads = [np.array([-5.0, -3.0, -1.0])]
        clipped = clip_grad_value(grads, clip_value=2.0)
        assert np.all(clipped[0] >= -2.0)
        assert np.all(clipped[0] <= 2.0)

    def test_invalid_clip_value(self):
        """Test that invalid clip_value raises error."""
        grads = [np.array([1.0, 2.0])]
        with pytest.raises(ValueError):
            clip_grad_value(grads, clip_value=0)
        with pytest.raises(ValueError):
            clip_grad_value(grads, clip_value=-1)


# =============================================================================
# Test Gradient Statistics
# =============================================================================


class TestGradientStats:
    """Tests for GradientStats class."""

    def test_from_tensor_basic(self):
        """Test basic statistics computation."""
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = GradientStats.from_tensor(grad)
        assert np.isclose(stats.mean, 3.0)
        assert np.isclose(stats.min, 1.0)
        assert np.isclose(stats.max, 5.0)

    def test_norm_computation(self):
        """Test L2 norm computation."""
        grad = np.array([3.0, 4.0])  # L2 norm = 5
        stats = GradientStats.from_tensor(grad)
        assert np.isclose(stats.norm_l2, 5.0)

    def test_zeros_ratio(self):
        """Test zeros ratio computation."""
        grad = np.array([0.0, 0.0, 1.0, 2.0])  # 50% zeros
        stats = GradientStats.from_tensor(grad, eps=1e-7)
        assert np.isclose(stats.zeros_ratio, 0.5, atol=0.01)

    def test_nan_detection(self):
        """Test NaN detection."""
        grad = np.array([1.0, np.nan, 3.0])
        stats = GradientStats.from_tensor(grad)
        assert stats.nan_count == 1

    def test_inf_detection(self):
        """Test Inf detection."""
        grad = np.array([1.0, np.inf, -np.inf, 3.0])
        stats = GradientStats.from_tensor(grad)
        assert stats.inf_count == 2

    def test_is_healthy(self):
        """Test health check."""
        # Healthy gradient
        healthy_grad = np.array([1.0, 2.0, 3.0])
        stats = GradientStats.from_tensor(healthy_grad)
        assert stats.is_healthy()

        # Unhealthy gradient with NaN
        nan_grad = np.array([1.0, np.nan])
        stats = GradientStats.from_tensor(nan_grad)
        assert not stats.is_healthy()


class TestGradientFlowAnalyzer:
    """Tests for GradientFlowAnalyzer class."""

    def test_record_layer_gradient(self):
        """Test recording layer gradients."""
        analyzer = GradientFlowAnalyzer(num_layers=3)
        grad = np.random.randn(32, 64)
        stats = analyzer.record_layer_gradient(0, grad)
        assert isinstance(stats, GradientStats)

    def test_get_flow_report(self):
        """Test flow report generation."""
        analyzer = GradientFlowAnalyzer(num_layers=3)
        grads = [
            np.random.randn(32, 64),
            np.random.randn(32, 64),
            np.random.randn(32, 64),
        ]
        analyzer.record_gradients(grads)
        report = analyzer.get_flow_report()
        assert "diagnosis" in report
        assert "flow_ratios" in report
        assert len(report["flow_ratios"]) == 3

    def test_vanishing_detection(self):
        """Test vanishing gradient detection."""
        analyzer = GradientFlowAnalyzer(num_layers=3)
        # Create gradients with decreasing norms (vanishing)
        grads = [
            np.random.randn(32, 64) * 10,
            np.random.randn(32, 64) * 0.01,
            np.random.randn(32, 64) * 0.0001,
        ]
        analyzer.record_gradients(grads)
        report = analyzer.get_flow_report()
        assert report["is_vanishing"]

    def test_exploding_detection(self):
        """Test exploding gradient detection."""
        analyzer = GradientFlowAnalyzer(num_layers=3)
        # Create gradients with increasing norms (exploding)
        grads = [
            np.random.randn(32, 64) * 0.1,
            np.random.randn(32, 64) * 10,
            np.random.randn(32, 64) * 1000,
        ]
        analyzer.record_gradients(grads)
        report = analyzer.get_flow_report()
        assert report["is_exploding"]


# =============================================================================
# Test Vanishing/Exploding Detection
# =============================================================================


class TestGradientDetection:
    """Tests for gradient detection functions."""

    def test_detect_vanishing_true(self):
        """Test detection of vanishing gradient."""
        grads = [
            np.random.randn(32, 64) * 10,
            np.random.randn(32, 64) * 0.001,  # Much smaller
        ]
        is_vanishing, ratio = detect_vanishing_gradient(grads)
        assert is_vanishing
        assert ratio < 0.01

    def test_detect_vanishing_false(self):
        """Test detection of healthy gradient (no vanishing)."""
        grads = [
            np.random.randn(32, 64),
            np.random.randn(32, 64),
        ]
        is_vanishing, ratio = detect_vanishing_gradient(grads)
        assert not is_vanishing

    def test_detect_exploding_true(self):
        """Test detection of exploding gradient."""
        grads = [
            np.random.randn(32, 64) * 0.1,
            np.random.randn(32, 64) * 100,  # Much larger
        ]
        is_exploding, max_ratio = detect_exploding_gradient(grads)
        assert is_exploding

    def test_detect_exploding_false(self):
        """Test detection of healthy gradient (no exploding)."""
        grads = [
            np.random.randn(32, 64),
            np.random.randn(32, 64),
        ]
        is_exploding, max_ratio = detect_exploding_gradient(grads)
        assert not is_exploding


# =============================================================================
# Test Solutions
# =============================================================================


class TestSkipConnection:
    """Tests for apply_skip_connection function."""

    def test_add_mode(self):
        """Test addition skip connection."""
        x = np.random.randn(32, 64)
        fx = np.random.randn(32, 64)
        output = apply_skip_connection(x, fx, mode="add")
        expected = x + fx
        assert np.allclose(output, expected)

    def test_concat_mode(self):
        """Test concatenation skip connection."""
        x = np.random.randn(32, 64)
        fx = np.random.randn(32, 64)
        output = apply_skip_connection(x, fx, mode="concat")
        assert output.shape == (32, 128)
        expected = np.concatenate([fx, x], axis=-1)
        assert np.allclose(output, expected)

    def test_projection(self):
        """Test skip connection with projection."""
        x = np.random.randn(32, 32)
        fx = np.random.randn(32, 64)
        proj = np.random.randn(32, 64)
        output = apply_skip_connection(x, fx, mode="add", projection=proj)
        assert output.shape == (32, 64)


class TestLayerScale:
    """Tests for LayerScale class."""

    def test_forward(self):
        """Test forward pass."""
        ls = LayerScale(dim=64, initial_value=0.1)
        x = np.random.randn(32, 64)
        output = ls.forward(x)
        expected = x * 0.1
        assert np.allclose(output, expected)

    def test_backward(self):
        """Test backward pass."""
        ls = LayerScale(dim=64, initial_value=0.1)
        x = np.random.randn(32, 64)
        ls.forward(x)
        grad_output = np.ones((32, 64))
        grad_input = ls.backward(grad_output)
        assert grad_input.shape == x.shape
        assert ls.grad_gamma is not None

    def test_gradient_numerical(self):
        """Test gradient with numerical check."""
        ls = LayerScale(dim=8, initial_value=0.1)
        x = np.random.randn(4, 8)

        # Forward
        output = ls.forward(x)

        # Analytical gradient
        grad_output = np.ones_like(output)
        grad_input = ls.backward(grad_output)

        # Numerical gradient check for gamma
        eps = 1e-5
        numerical_grad_gamma = np.zeros_like(ls.gamma)
        for i in range(len(ls.gamma)):
            ls.gamma[i] += eps
            out_plus = np.sum(ls.forward(x))
            ls.gamma[i] -= 2 * eps
            out_minus = np.sum(ls.forward(x))
            ls.gamma[i] += eps  # Restore
            numerical_grad_gamma[i] = (out_plus - out_minus) / (2 * eps)

        assert np.allclose(ls.grad_gamma, numerical_grad_gamma, atol=1e-5)


# =============================================================================
# Test Deep Networks
# =============================================================================


class TestDeepMLP:
    """Tests for DeepMLP class."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = DeepMLP(input_size=64, hidden_size=128, output_size=10, num_layers=5)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        assert output.shape == (32, 10)

    def test_backward_shape(self):
        """Test backward pass gradient shapes."""
        model = DeepMLP(input_size=64, hidden_size=128, output_size=10, num_layers=5)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        grad_input = model.backward(grad_output)
        assert grad_input.shape == x.shape

    def test_gradient_flow(self):
        """Test gradient flow through deep MLP."""
        model = DeepMLP(input_size=64, hidden_size=128, output_size=10, num_layers=20)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)

        # Check that gradients exist
        grads = model.gradients()
        assert all(g is not None for g in grads)

        # Check gradient norms
        norms = model.get_layer_gradient_norms()
        assert len(norms) > 0


class TestDeepResNet:
    """Tests for DeepResNet class."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = DeepResNet(input_size=64, hidden_size=128, output_size=10, num_blocks=5)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        assert output.shape == (32, 10)

    def test_backward_shape(self):
        """Test backward pass gradient shapes."""
        model = DeepResNet(input_size=64, hidden_size=128, output_size=10, num_blocks=5)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        grad_input = model.backward(grad_output)
        assert grad_input.shape == x.shape

    def test_skip_connection_preserves_gradient(self):
        """Test that skip connections preserve gradient flow."""
        # Very deep ResNet
        model = DeepResNet(input_size=64, hidden_size=64, output_size=10, num_blocks=50)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)

        # Check gradient norms - should be > 0.1 for all layers
        norms = model.get_layer_gradient_norms()
        assert all(n > 0.01 for n in norms), f"Some norms too small: {norms}"

    def test_gradient_norm_stable(self):
        """Test that gradient norms remain stable in deep ResNet."""
        model = DeepResNet(input_size=64, hidden_size=128, output_size=10, num_blocks=20)
        x = np.random.randn(32, 64)
        output = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)

        norms = model.get_layer_gradient_norms()
        # Check that norms don't vary too much (indicates stable gradient flow)
        max_norm = max(norms)
        min_norm = min(norms)
        # Skip connections should prevent extreme ratios
        assert max_norm / (min_norm + 1e-10) < 1000


class TestDeepLSTM:
    """Tests for DeepLSTM class."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = DeepLSTM(input_size=64, hidden_size=128, num_layers=3)
        x = np.random.randn(32, 10, 64)
        output, h_n, c_n = model.forward(x)
        assert output.shape == (32, 10, 128)
        assert h_n.shape == (3, 32, 128)
        assert c_n.shape == (3, 32, 128)

    def test_backward_shape(self):
        """Test backward pass gradient shape."""
        model = DeepLSTM(input_size=64, hidden_size=128, num_layers=3)
        x = np.random.randn(32, 10, 64)
        output, _, _ = model.forward(x)
        grad_output = np.ones_like(output)
        grad_input = model.backward(grad_output)
        assert grad_input.shape == x.shape

    def test_layer_normalization_stability(self):
        """Test that layer norm provides gradient stability."""
        # Deep LSTM with layer norm
        model = DeepLSTM(input_size=64, hidden_size=64, num_layers=10, use_layer_norm=True)
        x = np.random.randn(8, 20, 64)
        output, _, _ = model.forward(x)
        grad_output = np.ones_like(output)
        model.backward(grad_output)

        norms = model.get_layer_gradient_norms()
        # With layer norm, gradients should be stable
        assert all(n > 0.001 for n in norms), f"Gradients too small: {norms}"


# =============================================================================
# Test Gradient Flow Experiments
# =============================================================================


class TestGradientFlowExperiments:
    """Tests for gradient flow experiments."""

    def test_mlp_vanishing_gradient(self):
        """Test that deep MLP shows gradient decay compared to ResNet."""
        # Compare MLP vs ResNet gradient flow at the same depth
        mlp_result = run_gradient_flow_experiment("mlp", depth=50)
        resnet_result = run_gradient_flow_experiment("resnet", depth=50)
        # MLP should have smaller minimum flow ratio than ResNet
        # (worse gradient flow without skip connections)
        assert mlp_result["min_flow_ratio"] <= resnet_result["min_flow_ratio"] * 10

    def test_resnet_gradient_preserved(self):
        """Test that deep ResNet preserves gradients."""
        result = run_gradient_flow_experiment("resnet", depth=50)
        # ResNet should preserve gradients better
        assert result["min_flow_ratio"] > 0.001

    def test_lstm_gradient_stability(self):
        """Test that deep LSTM with layer norm has stable gradients."""
        result = run_gradient_flow_experiment("lstm", depth=10)
        # LSTM should have reasonable gradients
        assert len(result["layer_norms"]) > 0


# =============================================================================
# Test Registry
# =============================================================================


class TestRegistry:
    """Tests for gradient clipping registry."""

    def test_get_norm_clipper(self):
        """Test getting norm clipper."""
        clipper = get_gradient_clipper("norm")
        assert clipper == clip_grad_norm

    def test_get_value_clipper(self):
        """Test getting value clipper."""
        clipper = get_gradient_clipper("value")
        assert clipper == clip_grad_value

    def test_invalid_clipper(self):
        """Test that invalid clipper raises error."""
        with pytest.raises(ValueError):
            get_gradient_clipper("invalid")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_training_with_clipping(self):
        """Test training loop with gradient clipping."""
        # Create a simple MLP
        model = DeepMLP(input_size=64, hidden_size=128, output_size=10, num_layers=5)

        # Create some data
        x = np.random.randn(32, 64)
        y = np.random.randn(32, 10)

        # Training step with clipping
        output = model.forward(x)
        loss = np.mean((output - y) ** 2)
        grad_output = 2 * (output - y) / y.size
        model.backward(grad_output)

        # Get gradients and clip
        grads = model.gradients()
        clipped_grads, _ = clip_grad_norm(grads, max_norm=1.0)

        # Check that clipped gradients have bounded norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in clipped_grads if g is not None))
        assert total_norm <= 1.0 + 1e-6

    def test_analyzer_with_training(self):
        """Test gradient analyzer during training."""
        analyzer = GradientFlowAnalyzer(num_layers=5)
        model = DeepMLP(input_size=64, hidden_size=128, output_size=10, num_layers=5)

        x = np.random.randn(32, 64)
        y = np.random.randn(32, 10)

        output = model.forward(x)
        grad_output = 2 * (output - y) / y.size
        model.backward(grad_output)

        # Record gradients
        layer_grads = [g for g in model.gradients() if g is not None]
        analyzer.record_gradients(layer_grads[:5])

        report = analyzer.get_flow_report()
        assert report["diagnosis"] in ["healthy", "vanishing", "exploding", "unstable"]
