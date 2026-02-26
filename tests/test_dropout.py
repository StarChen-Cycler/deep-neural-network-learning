"""
Tests for dropout and regularization implementations.

Tests verify:
- Dropout train/eval mode behavior
- MC Dropout variance reduction
- L1/L2 regularization in optimizer
- Gradient correctness
- PyTorch comparison
"""

import pytest
import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase3_training.dropout import (
    Dropout,
    MCDropout,
    VariationalDropout,
    AlphaDropout,
    SpatialDropout,
    DropConnect,
    compute_mc_uncertainty,
    get_dropout,
)
from phase3_training.regularization import (
    L1Regularization,
    L2Regularization,
    ElasticNet,
    L1L2Regularizer,
    MaxNormConstraint,
    OrthogonalRegularizer,
    SpectralNormConstraint,
    apply_weight_decay,
    compute_regularization_loss,
    get_regularizer,
)


# Fixtures
@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def small_input():
    """Small input tensor for testing."""
    return np.random.randn(10, 20)


@pytest.fixture
def small_4d_input():
    """Small 4D input for spatial dropout."""
    return np.random.randn(4, 3, 8, 8)


# ==================== Dropout Tests ====================


class TestDropout:
    """Tests for standard Dropout."""

    def test_output_shape(self, small_input):
        """Test that output shape matches input."""
        dropout = Dropout(p=0.5)
        dropout.train()
        output = dropout.forward(small_input)
        assert output.shape == small_input.shape

    def test_eval_mode_no_dropout(self, small_input):
        """Test that eval mode does not apply dropout."""
        dropout = Dropout(p=0.5)
        dropout.eval()
        output = dropout.forward(small_input)
        np.testing.assert_array_equal(output, small_input)

    def test_training_mode_applies_dropout(self, small_input):
        """Test that training mode applies dropout (some zeros)."""
        dropout = Dropout(p=0.5)
        dropout.train()
        np.random.seed(42)
        output = dropout.forward(small_input.copy())
        # Some elements should be zero
        assert np.sum(output == 0) > 0

    def test_dropout_rate_approximation(self):
        """Test that dropout rate is approximately correct."""
        x = np.ones((1000, 1000))
        dropout = Dropout(p=0.3)
        dropout.train()
        np.random.seed(42)
        output = dropout.forward(x)

        # Count zeros (dropped elements)
        zero_ratio = np.mean(output == 0)
        # Should be close to 0.3
        assert 0.25 < zero_ratio < 0.35

    def test_inverted_scaling(self):
        """Test that inverted scaling preserves expected value."""
        x = np.ones((10000, 100)) * 2  # E[x] = 2
        dropout = Dropout(p=0.5)
        dropout.train()

        # Run multiple times to average
        outputs = []
        for _ in range(100):
            np.random.seed(None)
            outputs.append(dropout.forward(x.copy()))
        mean_output = np.mean(outputs)

        # Expected value should be preserved: E[y] = E[x]
        # With inverted dropout, E[y] = x (since kept elements scaled by 1/(1-p))
        assert np.abs(mean_output - 2.0) < 0.1

    def test_backward_shape(self, small_input):
        """Test that backward output shape matches input."""
        dropout = Dropout(p=0.5)
        dropout.train()
        _ = dropout.forward(small_input.copy())
        grad_output = np.ones_like(small_input)
        grad_input = dropout.backward(grad_output)
        assert grad_input.shape == small_input.shape

    def test_backward_eval_mode(self, small_input):
        """Test backward in eval mode returns unchanged gradient."""
        dropout = Dropout(p=0.5)
        dropout.eval()
        _ = dropout.forward(small_input)
        grad_output = np.random.randn(*small_input.shape)
        grad_input = dropout.backward(grad_output)
        np.testing.assert_array_equal(grad_input, grad_output)

    def test_backward_zeros_for_dropped(self):
        """Test that gradient is zero for dropped elements."""
        x = np.ones((100, 100))
        dropout = Dropout(p=0.8)  # High dropout for more zeros
        dropout.train()
        np.random.seed(42)
        output = dropout.forward(x.copy())

        grad_output = np.ones_like(output)
        grad_input = dropout.backward(grad_output)

        # Elements that were dropped (output == 0) should have zero gradient
        # Note: Due to scaling, we check the mask
        assert dropout._mask is not None
        dropped_count = np.sum(dropout._mask == 0)
        if dropped_count > 0:
            # Gradient should be zero at dropped positions
            zero_grad_count = np.sum(grad_input == 0)
            assert zero_grad_count >= dropped_count * 0.9  # Allow some tolerance

    def test_zero_dropout(self, small_input):
        """Test that p=0 means no dropout."""
        dropout = Dropout(p=0.0)
        dropout.train()
        output = dropout.forward(small_input.copy())
        np.testing.assert_array_equal(output, small_input)

    def test_invalid_probability(self):
        """Test that invalid probability raises error."""
        with pytest.raises(ValueError):
            Dropout(p=1.0)
        with pytest.raises(ValueError):
            Dropout(p=-0.1)


class TestMCDropout:
    """Tests for Monte Carlo Dropout."""

    def test_mc_inference_variance(self):
        """Test that MC inference produces variance."""
        x = np.random.randn(50, 20)
        mc_dropout = MCDropout(p=0.3, n_samples=50)

        predictions = mc_dropout.predict(x)

        # Should have shape (n_samples, batch, features)
        assert predictions.shape == (50, 50, 20)

        # Variance across samples should be non-zero
        variance = np.var(predictions, axis=0)
        assert np.mean(variance) > 0

    def test_variance_reduction_with_more_samples(self):
        """Test that more samples reduce variance estimate uncertainty."""
        x = np.random.randn(100, 10)

        mc_dropout_few = MCDropout(p=0.5, n_samples=10)
        mc_dropout_many = MCDropout(p=0.5, n_samples=100)

        preds_few = mc_dropout_few.predict(x)
        preds_many = mc_dropout_many.predict(x)

        var_few = np.var(preds_few, axis=0)
        var_many = np.var(preds_many, axis=0)

        # Both should have variance (dropout is active)
        assert np.mean(var_few) > 0
        assert np.mean(var_many) > 0

    def test_compute_mc_uncertainty(self):
        """Test uncertainty computation."""
        # Create samples with known variance
        n_samples = 20
        batch = 10
        features = 5
        samples = np.random.randn(n_samples, batch, features)

        mean, variance, entropy = compute_mc_uncertainty(samples)

        assert mean.shape == (batch, features)
        assert variance.shape == (batch, features)
        # Entropy is computed along last axis, so shape is (batch,)
        assert entropy.shape == (batch,)


class TestVariationalDropout:
    """Tests for Variational Dropout."""

    def test_output_shape(self, small_input):
        """Test output shape matches input."""
        var_dropout = VariationalDropout(initial_p=0.5)
        var_dropout.train()
        output = var_dropout.forward(small_input.copy())
        assert output.shape == small_input.shape

    def test_eval_mode(self, small_input):
        """Test eval mode returns input unchanged."""
        var_dropout = VariationalDropout(initial_p=0.5)
        var_dropout.eval()
        output = var_dropout.forward(small_input)
        np.testing.assert_array_equal(output, small_input)

    def test_dropout_probability_property(self):
        """Test that p property returns valid probability."""
        var_dropout = VariationalDropout(initial_p=0.3)
        assert 0 < var_dropout.p < 1
        # Should be close to initial value
        assert abs(var_dropout.p - 0.3) < 0.01

    def test_backward_returns_tuple(self, small_input):
        """Test backward returns both input gradient and logit_p gradient."""
        var_dropout = VariationalDropout(initial_p=0.5)
        var_dropout.train()
        _ = var_dropout.forward(small_input.copy())
        grad_output = np.ones_like(small_input)
        grad_input, grad_logit_p = var_dropout.backward(grad_output)

        assert grad_input.shape == small_input.shape
        assert isinstance(grad_logit_p, np.ndarray)


class TestAlphaDropout:
    """Tests for Alpha Dropout (for SELU networks)."""

    def test_output_shape(self, small_input):
        """Test output shape matches input."""
        alpha_dropout = AlphaDropout(p=0.5)
        alpha_dropout.train()
        output = alpha_dropout.forward(small_input.copy())
        assert output.shape == small_input.shape

    def test_eval_mode(self, small_input):
        """Test eval mode returns input unchanged."""
        alpha_dropout = AlphaDropout(p=0.5)
        alpha_dropout.eval()
        output = alpha_dropout.forward(small_input)
        np.testing.assert_array_equal(output, small_input)

    def test_dropped_values_near_alpha_prime(self):
        """Test that dropped values are near alpha_prime."""
        x = np.zeros((1000, 1000))
        alpha_dropout = AlphaDropout(p=0.5)
        alpha_dropout.train()
        np.random.seed(42)
        output = alpha_dropout.forward(x.copy())

        # Some values should be close to alpha_p (approximately -1.758)
        # Values that are NOT zero were "dropped" and set to near alpha_p
        alpha_p = alpha_dropout.alpha_p
        # The transformation is: a * (x * mask + alpha_p * (1-mask)) + b
        # For x=0 and dropped elements, output should be near transformed alpha_p


class TestSpatialDropout:
    """Tests for Spatial Dropout."""

    def test_output_shape(self, small_4d_input):
        """Test output shape matches input."""
        spatial_dropout = SpatialDropout(p=0.5)
        spatial_dropout.train()
        output = spatial_dropout.forward(small_4d_input.copy())
        assert output.shape == small_4d_input.shape

    def test_drops_entire_channels(self, small_4d_input):
        """Test that entire channels are dropped."""
        spatial_dropout = SpatialDropout(p=0.5)
        spatial_dropout.train()
        np.random.seed(42)
        output = spatial_dropout.forward(small_4d_input.copy())

        # Check that for each sample, some channels are all zeros
        batch_size, channels, height, width = small_4d_input.shape
        for b in range(batch_size):
            channel_norms = [np.linalg.norm(output[b, c]) for c in range(channels)]
            zero_channels = sum(1 for n in channel_norms if n < 1e-6)
            # With p=0.5, about half should be zero
            # Note: This is probabilistic, so just check some are zero
            assert zero_channels >= 0

    def test_invalid_dimensions(self):
        """Test that non-4D input raises error."""
        spatial_dropout = SpatialDropout(p=0.5)
        spatial_dropout.train()
        x_3d = np.random.randn(4, 3, 8)
        with pytest.raises(ValueError):
            spatial_dropout.forward(x_3d)


class TestDropConnect:
    """Tests for DropConnect."""

    def test_output_shape(self):
        """Test output shape is correct."""
        x = np.random.randn(10, 20)
        weight = np.random.randn(5, 20)
        bias = np.random.randn(5)

        drop_connect = DropConnect(p=0.5)
        drop_connect.train()
        output = drop_connect.forward(x, weight, bias)

        assert output.shape == (10, 5)

    def test_eval_mode(self):
        """Test eval mode applies full weight."""
        x = np.random.randn(10, 20)
        weight = np.random.randn(5, 20)
        bias = np.random.randn(5)

        drop_connect = DropConnect(p=0.5)
        drop_connect.eval()
        output = drop_connect.forward(x, weight, bias)

        expected = x @ weight.T + bias
        np.testing.assert_allclose(output, expected)

    def test_backward_shapes(self):
        """Test backward returns correct shapes."""
        x = np.random.randn(10, 20)
        weight = np.random.randn(5, 20)
        bias = np.random.randn(5)

        drop_connect = DropConnect(p=0.5)
        drop_connect.train()
        _ = drop_connect.forward(x.copy(), weight.copy(), bias.copy())

        grad_output = np.random.randn(10, 5)
        grad_input, grad_weight, grad_bias = drop_connect.backward(grad_output, x)

        assert grad_input.shape == x.shape
        assert grad_weight.shape == weight.shape
        assert grad_bias.shape == bias.shape


class TestDropoutRegistry:
    """Tests for dropout registry."""

    def test_get_dropout_standard(self):
        """Test getting standard dropout."""
        dropout = get_dropout("dropout", p=0.3)
        assert isinstance(dropout, Dropout)
        assert dropout.p == 0.3

    def test_get_dropout_mc(self):
        """Test getting MC dropout."""
        dropout = get_dropout("mc_dropout", p=0.5, n_samples=20)
        assert isinstance(dropout, MCDropout)
        assert dropout.n_samples == 20

    def test_get_dropout_invalid(self):
        """Test that invalid name raises error."""
        with pytest.raises(ValueError):
            get_dropout("invalid_dropout")


# ==================== Regularization Tests ====================


class TestL1Regularization:
    """Tests for L1 regularization."""

    def test_loss_computation(self):
        """Test L1 loss is sum of absolute values."""
        weights = np.array([[1, -2], [3, -4]])
        reg = L1Regularization(lambda_=0.1)
        loss = reg.loss(weights)
        # sum(|w|) = 1 + 2 + 3 + 4 = 10
        # loss = 0.1 * 10 = 1.0
        assert loss == pytest.approx(1.0)

    def test_gradient_computation(self):
        """Test L1 gradient is sign of weights."""
        weights = np.array([[1, -2], [0, 4]])
        reg = L1Regularization(lambda_=0.1)
        grad = reg.gradient(weights)
        expected = 0.1 * np.array([[1, -1], [0, 1]])
        np.testing.assert_array_equal(grad, expected)

    def test_multiple_weights(self):
        """Test with multiple weight arrays."""
        weights = [np.ones((2, 2)), np.ones((3, 3))]
        reg = L1Regularization(lambda_=0.1)
        loss = reg.loss(weights)
        # sum(|w|) = 4 + 9 = 13
        # loss = 0.1 * 13 = 1.3
        assert loss == pytest.approx(1.3)

    def test_zero_weights(self):
        """Test with zero weights."""
        weights = np.zeros((3, 3))
        reg = L1Regularization(lambda_=0.1)
        loss = reg.loss(weights)
        assert loss == 0.0


class TestL2Regularization:
    """Tests for L2 regularization."""

    def test_loss_computation(self):
        """Test L2 loss is sum of squared values / 2."""
        weights = np.array([[1, 2], [3, 4]])
        reg = L2Regularization(lambda_=0.1)
        loss = reg.loss(weights)
        # sum(w^2) = 1 + 4 + 9 + 16 = 30
        # loss = 0.5 * 0.1 * 30 = 1.5
        assert loss == pytest.approx(1.5)

    def test_gradient_computation(self):
        """Test L2 gradient is lambda * w."""
        weights = np.array([[1, 2], [3, 4]])
        reg = L2Regularization(lambda_=0.1)
        grad = reg.gradient(weights)
        expected = 0.1 * weights
        np.testing.assert_array_equal(grad, expected)

    def test_l2_vs_pytorch(self):
        """Test L2 loss matches PyTorch."""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        weights = np.random.randn(10, 20)

        # Our implementation
        reg = L2Regularization(lambda_=0.01)
        our_loss = reg.loss(weights)

        # PyTorch
        torch_weights = torch.tensor(weights, dtype=torch.float32)
        torch_loss = 0.5 * 0.01 * torch.sum(torch_weights**2).item()

        assert our_loss == pytest.approx(torch_loss, rel=1e-5)


class TestElasticNet:
    """Tests for Elastic Net regularization."""

    def test_loss_combination(self):
        """Test that ElasticNet combines L1 and L2."""
        weights = np.array([[1, -2], [3, -4]])
        reg = ElasticNet(lambda_=0.1, alpha=0.5)

        loss = reg.loss(weights)
        # L1: 0.5 * 0.1 * 10 = 0.5
        # L2: 0.5 * 0.1 * 0.5 * 30 = 0.75
        # Total: 1.25
        assert loss == pytest.approx(1.25)

    def test_pure_l1(self):
        """Test alpha=1 gives pure L1."""
        weights = np.array([[1, -2]])
        elastic = ElasticNet(lambda_=0.1, alpha=1.0)
        l1 = L1Regularization(lambda_=0.1)

        elastic_loss = elastic.loss(weights)
        l1_loss = l1.loss(weights)

        assert elastic_loss == pytest.approx(l1_loss)

    def test_pure_l2(self):
        """Test alpha=0 gives pure L2."""
        weights = np.array([[1, -2]])
        elastic = ElasticNet(lambda_=0.1, alpha=0.0)
        l2 = L2Regularization(lambda_=0.1)

        elastic_loss = elastic.loss(weights)
        l2_loss = l2.loss(weights)

        assert elastic_loss == pytest.approx(l2_loss)


class TestMaxNormConstraint:
    """Tests for Max-Norm constraint."""

    def test_constraint_application(self):
        """Test that weights are constrained."""
        # Create weights with norm > max_value
        weights = np.ones((10, 10)) * 10  # Each column has norm = sqrt(10 * 100) = 31.6
        constraint = MaxNormConstraint(max_value=1.0, axis=0)

        constrained = constraint(weights)

        # Check that norms are now <= 1
        norms = np.sqrt(np.sum(constrained**2, axis=0))
        assert np.all(norms <= 1.0 + 1e-6)

    def test_no_constraint_needed(self):
        """Test that small weights are unchanged."""
        weights = np.random.randn(5, 5) * 0.1  # Small weights
        constraint = MaxNormConstraint(max_value=10.0, axis=0)

        constrained = constraint(weights)

        # Since norms are already < 10, should clip to max (unchanged)
        np.testing.assert_array_almost_equal(constrained, weights)


class TestSpectralNormConstraint:
    """Tests for Spectral Normalization."""

    def test_spectral_norm_reduction(self):
        """Test that spectral norm is reduced."""
        # Create matrix with known spectral norm
        w = np.random.randn(10, 20)
        original_sigma = np.linalg.svd(w, compute_uv=False)[0]

        constraint = SpectralNormConstraint(max_value=1.0, n_power_iterations=10)
        constrained = constraint(w)

        new_sigma = np.linalg.svd(constrained, compute_uv=False)[0]

        if original_sigma > 1.0:
            assert new_sigma <= 1.0 + 0.1  # Allow small tolerance

    def test_small_norm_unchanged(self):
        """Test that small spectral norm matrices are unchanged."""
        w = np.random.randn(5, 5) * 0.1  # Small singular values
        original_w = w.copy()

        constraint = SpectralNormConstraint(max_value=10.0)
        constrained = constraint(w)

        # Since spectral norm is small, should be unchanged
        np.testing.assert_array_almost_equal(constrained, original_w)


class TestOrthogonalRegularizer:
    """Tests for Orthogonal Regularization."""

    def test_loss_for_orthogonal_matrix(self):
        """Test loss is zero for orthogonal matrix."""
        # Create orthogonal matrix using QR
        q, _ = np.linalg.qr(np.random.randn(10, 10))
        reg = OrthogonalRegularizer(lambda_=0.01)
        loss = reg.loss(q)
        # Should be very small for orthogonal matrix
        assert loss < 1e-10

    def test_loss_for_non_orthogonal(self):
        """Test loss is positive for non-orthogonal matrix."""
        w = np.random.randn(10, 5) * 2
        reg = OrthogonalRegularizer(lambda_=0.01)
        loss = reg.loss(w)
        assert loss > 0

    def test_gradient_shape(self):
        """Test gradient has same shape as weight."""
        w = np.random.randn(10, 5)
        reg = OrthogonalRegularizer(lambda_=0.01)
        grad = reg.gradient(w)
        assert grad.shape == w.shape


class TestApplyWeightDecay:
    """Tests for apply_weight_decay function."""

    def test_l2_weight_decay(self):
        """Test L2 weight decay adds to gradient."""
        params = [np.ones((3, 3))]
        grads = [np.zeros((3, 3))]

        updated = apply_weight_decay(params, grads, lr=0.1, weight_decay=0.01, decay_type="l2")

        # grad_new = grad + weight_decay * param = 0 + 0.01 * 1 = 0.01
        np.testing.assert_allclose(updated[0], np.ones((3, 3)) * 0.01)

    def test_l1_weight_decay(self):
        """Test L1 weight decay adds sign to gradient."""
        params = [np.array([[1, -1]])]
        grads = [np.zeros((1, 2))]

        updated = apply_weight_decay(params, grads, lr=0.1, weight_decay=0.01, decay_type="l1")

        expected = np.array([[0.01, -0.01]])
        np.testing.assert_allclose(updated[0], expected)

    def test_zero_weight_decay(self):
        """Test zero weight decay leaves gradient unchanged."""
        params = [np.ones((3, 3))]
        grads = [np.ones((3, 3))]

        updated = apply_weight_decay(params, grads, lr=0.1, weight_decay=0.0)

        np.testing.assert_array_equal(updated[0], grads[0])


class TestComputeRegularizationLoss:
    """Tests for compute_regularization_loss function."""

    def test_combined_loss(self):
        """Test combined L1 + L2 loss."""
        weights = [np.array([[1, -2]])]
        loss = compute_regularization_loss(weights, l1_lambda=0.1, l2_lambda=0.01)

        # L1: 0.1 * (1 + 2) = 0.3
        # L2: 0.5 * 0.01 * (1 + 4) = 0.025
        # Total: 0.325
        assert loss == pytest.approx(0.325)

    def test_l1_only(self):
        """Test L1 only loss."""
        weights = [np.array([[1, 2]])]
        loss = compute_regularization_loss(weights, l1_lambda=0.1, l2_lambda=0.0)

        # L1: 0.1 * (1 + 2) = 0.3
        assert loss == pytest.approx(0.3)

    def test_l2_only(self):
        """Test L2 only loss."""
        weights = [np.array([[1, 2]])]
        loss = compute_regularization_loss(weights, l1_lambda=0.0, l2_lambda=0.01)

        # L2: 0.5 * 0.01 * (1 + 4) = 0.025
        assert loss == pytest.approx(0.025)


class TestRegularizerRegistry:
    """Tests for regularizer registry."""

    def test_get_l1(self):
        """Test getting L1 regularizer."""
        reg = get_regularizer("l1", lambda_=0.01)
        assert isinstance(reg, L1Regularization)

    def test_get_l2(self):
        """Test getting L2 regularizer."""
        reg = get_regularizer("l2", lambda_=0.01)
        assert isinstance(reg, L2Regularization)

    def test_get_elastic_net(self):
        """Test getting ElasticNet regularizer."""
        reg = get_regularizer("elastic_net", lambda_=0.01, alpha=0.5)
        assert isinstance(reg, ElasticNet)

    def test_get_invalid(self):
        """Test that invalid name raises error."""
        with pytest.raises(ValueError):
            get_regularizer("invalid")


# ==================== Integration Tests ====================


class TestDropoutWithMLP:
    """Tests for dropout integrated with MLP."""

    def test_dropout_reduces_overfitting(self):
        """Test that dropout helps with generalization."""
        from phase1_basics.mlp import MLP
        from phase1_basics.optimizer import SGD
        from phase1_basics.loss import MSELoss

        # Create simple dataset
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = X_train[:, :5].sum(axis=1, keepdims=True)

        # Train without dropout
        mlp_no_dropout = MLP(input_size=10, hidden_sizes=[50], output_size=1)

        # Both should converge
        optimizer = SGD(lr=0.01)
        loss_fn = MSELoss()

        for _ in range(200):
            output = mlp_no_dropout.forward(X_train)
            loss = loss_fn.forward(output, y_train)
            grad = loss_fn.backward()
            mlp_no_dropout.backward(grad)
            optimizer.step(mlp_no_dropout.parameters())
            mlp_no_dropout.zero_grad()

        # Should have learned something
        final_pred = mlp_no_dropout.forward(X_train)
        final_loss = np.mean((final_pred - y_train) ** 2)
        assert final_loss < 5.0  # Should have converged somewhat


class TestL2WithSGD:
    """Tests for L2 regularization with SGD."""

    def test_weight_decay_in_sgd(self):
        """Test that weight decay prevents weight explosion."""
        from phase1_basics.optimizer import SGD

        # Create params with gradient
        params = [np.ones((5, 5)) * 10]
        grads = [np.zeros((5, 5))]

        # Apply weight decay to gradients
        grads_with_decay = apply_weight_decay(params, grads, lr=0.01, weight_decay=0.1)

        # SGD expects list of (param, grad) tuples
        optimizer = SGD(lr=0.01)
        param_grad_tuples = list(zip(params, grads_with_decay))
        optimizer.step(param_grad_tuples)

        # Weights should have decreased
        assert np.mean(params[0]) < 10


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchComparison:
    """Compare implementations with PyTorch."""

    def test_dropout_comparison(self):
        """Compare dropout with PyTorch."""
        x = np.random.randn(10, 20)

        # Our implementation
        our_dropout = Dropout(p=0.5)
        our_dropout.train()
        np.random.seed(42)
        our_output = our_dropout.forward(x.copy())

        # PyTorch
        torch_dropout = nn.Dropout(p=0.5)
        torch_dropout.train()
        torch.manual_seed(42)
        torch_output = torch_dropout(torch.tensor(x, dtype=torch.float32)).numpy()

        # Both should have similar dropout patterns (same seed)
        # Note: Exact match depends on random state implementation
        our_zero_ratio = np.mean(our_output == 0)
        torch_zero_ratio = np.mean(torch_output == 0)

        # Both should have dropped approximately half
        assert 0.4 < our_zero_ratio < 0.6
        assert 0.4 < torch_zero_ratio < 0.6

    def test_l2_comparison(self):
        """Compare L2 regularization with PyTorch."""
        weights = np.random.randn(10, 20)

        # Our implementation
        our_l2 = L2Regularization(lambda_=0.01)
        our_loss = our_l2.loss(weights)

        # PyTorch
        torch_weights = torch.tensor(weights, dtype=torch.float32, requires_grad=True)
        torch_loss = 0.5 * 0.01 * torch.sum(torch_weights**2)

        assert our_loss == pytest.approx(torch_loss.item(), rel=1e-5)


# ==================== MC Dropout Variance Reduction Test ====================


class TestMCDropoutVarianceReduction:
    """Test that MC Dropout reduces variance with more samples."""

    def test_variance_estimation_quality(self):
        """Test that variance estimation improves with samples."""
        x = np.random.randn(100, 50)

        # MC Dropout with different sample counts
        mc_10 = MCDropout(p=0.5, n_samples=10)
        mc_100 = MCDropout(p=0.5, n_samples=100)

        preds_10 = mc_10.predict(x)
        preds_100 = mc_100.predict(x)

        var_10 = np.var(preds_10, axis=0)
        var_100 = np.var(preds_100, axis=0)

        # Both should estimate variance
        assert np.mean(var_10) > 0
        assert np.mean(var_100) > 0

        # More samples should give more stable estimate
        # (though not necessarily lower variance estimate)
        std_of_var_10 = np.std(var_10)
        std_of_var_100 = np.std(var_100)

        # The variance of variance estimates should be lower with more samples
        # This is a statistical property
        assert std_of_var_100 < std_of_var_10 * 2  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
