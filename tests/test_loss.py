"""
Unit tests for loss functions.

Tests verify:
1. Forward pass correctness
2. Gradient correctness via numerical comparison
3. PyTorch comparison (when available)
4. Edge cases and numerical stability

Run: pytest tests/test_loss.py -v
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

from phase1_basics.loss import (
    MSELoss,
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
    TripletLoss,
    numerical_gradient_loss,
    numerical_gradient_triplet,
    get_loss,
    _softmax,
    _log_softmax,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def small_batch():
    """Small batch for quick tests."""
    return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.5, 2.5], [2.5, 3.5]])


@pytest.fixture
def classification_batch():
    """Batch for classification tests."""
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    targets = np.array([2, 1, 0])
    return logits, targets


# =============================================================================
# Test MSELoss
# =============================================================================


class TestMSELoss:
    """Test suite for MSE loss."""

    def test_forward_shape(self, small_batch):
        """Test that forward returns scalar."""
        pred, target = small_batch
        loss_fn = MSELoss()
        loss = loss_fn.forward(pred, target)
        assert np.isscalar(loss) or loss.shape == ()

    def test_forward_values(self):
        """Test MSE calculation correctness."""
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        loss_fn = MSELoss()
        loss = loss_fn.forward(pred, target)
        assert np.isclose(loss, 0.0)

    def test_forward_nonzero(self):
        """Test MSE with non-zero difference."""
        pred = np.array([[1.0], [2.0]])
        target = np.array([[2.0], [3.0]])
        loss_fn = MSELoss()
        loss = loss_fn.forward(pred, target)
        # MSE = mean((1-2)^2 + (2-3)^2) = mean(1 + 1) = 1
        assert np.isclose(loss, 1.0)

    def test_backward_shape(self, small_batch):
        """Test that backward returns same shape as input."""
        pred, target = small_batch
        loss_fn = MSELoss()
        loss_fn.forward(pred, target)
        grad = loss_fn.backward()
        assert grad.shape == pred.shape

    def test_backward_gradient_numerical(self, random_seed):
        """Test MSE gradient with numerical comparison."""
        pred = np.random.randn(4, 3)
        target = np.random.randn(4, 3)

        loss_fn = MSELoss()
        loss_fn.forward(pred, target)
        analytical = loss_fn.backward()

        numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

        assert np.allclose(analytical, numerical, atol=1e-6)

    def test_reduction_sum(self):
        """Test sum reduction."""
        pred = np.array([[1.0], [2.0]])
        target = np.array([[2.0], [3.0]])
        loss_fn = MSELoss(reduction="sum")
        loss = loss_fn.forward(pred, target)
        # Sum: (1-2)^2 + (2-3)^2 = 1 + 1 = 2
        assert np.isclose(loss, 2.0)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare with PyTorch MSE loss."""
        pred = np.random.randn(4, 3)
        target = np.random.randn(4, 3)

        # NumPy
        loss_fn = MSELoss()
        np_loss = loss_fn.forward(pred, target)
        np_grad = loss_fn.backward()

        # PyTorch
        pred_t = torch.tensor(pred, dtype=torch.float64, requires_grad=True)
        target_t = torch.tensor(target, dtype=torch.float64)
        torch_loss = nn.MSELoss()(pred_t, target_t)
        torch_loss.backward()

        assert np.isclose(np_loss, torch_loss.item(), atol=1e-6)
        assert np.allclose(np_grad, pred_t.grad.numpy(), atol=1e-6)


# =============================================================================
# Test CrossEntropyLoss
# =============================================================================


class TestCrossEntropyLoss:
    """Test suite for cross-entropy loss."""

    def test_forward_shape(self, classification_batch):
        """Test that forward returns scalar."""
        logits, targets = classification_batch
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)
        assert np.isscalar(loss) or loss.shape == ()

    def test_forward_perfect_prediction(self):
        """Test with perfect predictions."""
        # Perfect prediction: logits are very high for correct class
        logits = np.array([[0.0, 0.0, 100.0]])
        targets = np.array([2])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)
        # Should be close to 0
        assert loss < 0.01

    def test_backward_shape(self, classification_batch):
        """Test backward returns same shape as logits."""
        logits, targets = classification_batch
        loss_fn = CrossEntropyLoss()
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        assert grad.shape == logits.shape

    def test_backward_gradient_numerical(self, random_seed):
        """Test cross-entropy gradient numerically."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)

        loss_fn = CrossEntropyLoss()
        loss_fn.forward(logits, targets)
        analytical = loss_fn.backward()

        numerical = numerical_gradient_loss(loss_fn, logits.copy(), targets)

        assert np.allclose(analytical, numerical, atol=1e-6)

    def test_label_smoothing(self):
        """Test label smoothing modifies targets."""
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])

        loss_fn = CrossEntropyLoss(label_smoothing=0.1)
        loss_fn.forward(logits, targets)

        # Check smoothed targets
        expected = np.array([[0.0333, 0.0333, 0.9333]])
        assert np.allclose(loss_fn.target_one_hot, expected, atol=1e-3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare with PyTorch cross-entropy loss."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)

        # NumPy
        loss_fn = CrossEntropyLoss()
        np_loss = loss_fn.forward(logits, targets)
        np_grad = loss_fn.backward()

        # PyTorch
        logits_t = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        targets_t = torch.tensor(targets, dtype=torch.long)
        torch_loss = nn.CrossEntropyLoss()(logits_t, targets_t)
        torch_loss.backward()

        assert np.isclose(np_loss, torch_loss.item(), atol=1e-6)
        assert np.allclose(np_grad, logits_t.grad.numpy(), atol=1e-5)


# =============================================================================
# Test FocalLoss
# =============================================================================


class TestFocalLoss:
    """Test suite for focal loss."""

    def test_forward_shape(self, classification_batch):
        """Test that forward returns scalar."""
        logits, targets = classification_batch
        loss_fn = FocalLoss()
        loss = loss_fn.forward(logits, targets)
        assert np.isscalar(loss) or loss.shape == ()

    def test_gamma_zero_equals_ce(self, classification_batch):
        """With gamma=0, focal loss equals cross-entropy (scaled by alpha)."""
        logits, targets = classification_batch

        focal_fn = FocalLoss(gamma=0.0, alpha=1.0)
        focal_loss = focal_fn.forward(logits.copy(), targets)

        ce_fn = CrossEntropyLoss()
        ce_loss = ce_fn.forward(logits.copy(), targets)

        assert np.isclose(focal_loss, ce_loss, atol=1e-5)

    def test_high_gamma_reduces_easy_loss(self):
        """High gamma should reduce loss for easy (correct) predictions."""
        # Easy prediction: high logit for correct class
        easy_logits = np.array([[0.0, 0.0, 10.0]])
        targets = np.array([2])

        loss_low_gamma = FocalLoss(gamma=0.5, alpha=1.0).forward(easy_logits.copy(), targets)
        loss_high_gamma = FocalLoss(gamma=2.0, alpha=1.0).forward(easy_logits.copy(), targets)

        # Higher gamma should give lower loss for easy examples
        assert loss_high_gamma < loss_low_gamma

    def test_backward_shape(self, classification_batch):
        """Test backward returns same shape as logits."""
        logits, targets = classification_batch
        loss_fn = FocalLoss()
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        assert grad.shape == logits.shape

    def test_backward_gradient_numerical(self, random_seed):
        """Test focal loss gradient numerically."""
        logits = np.random.randn(3, 4)
        targets = np.random.randint(0, 4, size=3)

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        loss_fn.forward(logits, targets)
        analytical = loss_fn.backward()

        numerical = numerical_gradient_loss(loss_fn, logits.copy(), targets)

        # Focal loss gradient is more complex, allow slightly higher tolerance
        assert np.allclose(analytical, numerical, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare with PyTorch focal loss implementation."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)

        gamma = 2.0
        alpha = 0.25

        # NumPy
        loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
        np_loss = loss_fn.forward(logits.copy(), targets)

        # PyTorch focal loss (manual implementation)
        logits_t = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        targets_t = torch.tensor(targets, dtype=torch.long)

        ce_loss = F.cross_entropy(logits_t, targets_t, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        torch_loss = focal_loss.mean()

        assert np.isclose(np_loss, torch_loss.item(), atol=1e-5)


# =============================================================================
# Test LabelSmoothingLoss
# =============================================================================


class TestLabelSmoothingLoss:
    """Test suite for label smoothing loss."""

    def test_forward_shape(self, classification_batch):
        """Test that forward returns scalar."""
        logits, targets = classification_batch
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        loss = loss_fn.forward(logits, targets)
        assert np.isscalar(loss) or loss.shape == ()

    def test_smoothed_targets(self):
        """Test that targets are correctly smoothed."""
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])

        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        loss_fn.forward(logits, targets)

        # With epsilon=0.1, K=3:
        # true class: 1 - 0.1 + 0.1/3 = 0.9333
        # other classes: 0.1/3 = 0.0333
        expected = np.array([[0.0333, 0.0333, 0.9333]])
        assert np.allclose(loss_fn.smooth_target, expected, atol=1e-3)

    def test_backward_shape(self, classification_batch):
        """Test backward returns same shape as logits."""
        logits, targets = classification_batch
        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        assert grad.shape == logits.shape

    def test_backward_gradient_numerical(self, random_seed):
        """Test label smoothing gradient numerically."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)

        loss_fn = LabelSmoothingLoss(epsilon=0.1)
        loss_fn.forward(logits, targets)
        analytical = loss_fn.backward()

        numerical = numerical_gradient_loss(loss_fn, logits.copy(), targets)

        assert np.allclose(analytical, numerical, atol=1e-6)

    def test_epsilon_zero_equals_ce(self, classification_batch):
        """With epsilon=0, should equal regular cross-entropy."""
        logits, targets = classification_batch

        ls_fn = LabelSmoothingLoss(epsilon=0.0)
        ls_loss = ls_fn.forward(logits.copy(), targets)

        ce_fn = CrossEntropyLoss()
        ce_loss = ce_fn.forward(logits.copy(), targets)

        assert np.isclose(ls_loss, ce_loss, atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare with PyTorch label smoothing."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)
        epsilon = 0.1

        # NumPy
        loss_fn = LabelSmoothingLoss(epsilon=epsilon)
        np_loss = loss_fn.forward(logits.copy(), targets)

        # PyTorch
        logits_t = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
        targets_t = torch.tensor(targets, dtype=torch.long)
        torch_loss = F.cross_entropy(logits_t, targets_t, label_smoothing=epsilon)

        assert np.isclose(np_loss, torch_loss.item(), atol=1e-5)


# =============================================================================
# Test TripletLoss
# =============================================================================


class TestTripletLoss:
    """Test suite for triplet loss."""

    def test_forward_shape(self):
        """Test that forward returns scalar."""
        anchor = np.array([[1.0, 2.0]])
        positive = np.array([[1.1, 2.1]])
        negative = np.array([[5.0, 5.0]])

        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn.forward(anchor, positive, negative)
        assert np.isscalar(loss) or loss.shape == ()

    def test_easy_triplet_zero_loss(self):
        """Easy triplet (negative far away) should give zero loss."""
        anchor = np.array([[0.0, 0.0]])
        positive = np.array([[0.1, 0.1]])
        negative = np.array([[10.0, 10.0]])

        loss_fn = TripletLoss(margin=1.0)
        loss = loss_fn.forward(anchor, positive, negative)

        # d(a,p) ~ 0.14, d(a,n) ~ 14.14, margin = 1.0
        # max(0.14 - 14.14 + 1.0, 0) = 0
        assert loss < 0.01

    def test_hard_triplet_positive_loss(self):
        """Hard triplet (negative closer than positive) should give positive loss."""
        anchor = np.array([[0.0, 0.0]])
        positive = np.array([[1.0, 0.0]])
        negative = np.array([[0.5, 0.0]])

        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn.forward(anchor, positive, negative)

        # d(a,p) = 1.0, d(a,n) = 0.5, margin = 0.5
        # max(1.0 - 0.5 + 0.5, 0) = 1.0
        assert np.isclose(loss, 1.0, atol=0.1)

    def test_margin_constraint(self):
        """Test margin constraint: d(a,p) - d(a,n) + margin should be > 0 for hard triplets."""
        anchor = np.array([[0.0, 0.0]])
        positive = np.array([[2.0, 0.0]])  # d(a,p) = 2.0
        negative = np.array([[1.0, 0.0]])  # d(a,n) = 1.0

        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn.forward(anchor, positive, negative)

        # Loss = max(2.0 - 1.0 + 0.5, 0) = 1.5
        # This means: d(a,n) should be at least d(a,p) + margin = 2.5 for zero loss
        assert np.isclose(loss, 1.5, atol=0.1)

    def test_backward_shapes(self):
        """Test backward returns three gradients with correct shapes."""
        anchor = np.array([[1.0, 2.0], [3.0, 4.0]])
        positive = np.array([[1.1, 2.1], [3.1, 4.1]])
        negative = np.array([[5.0, 5.0], [6.0, 6.0]])

        loss_fn = TripletLoss(margin=0.5)
        loss_fn.forward(anchor, positive, negative)
        grad_a, grad_p, grad_n = loss_fn.backward()

        assert grad_a.shape == anchor.shape
        assert grad_p.shape == positive.shape
        assert grad_n.shape == negative.shape

    def test_backward_gradient_numerical(self, random_seed):
        """Test triplet loss gradient numerically."""
        anchor = np.random.randn(2, 3)
        positive = np.random.randn(2, 3)
        negative = np.random.randn(2, 3) + 2.0  # Make negatives different

        loss_fn = TripletLoss(margin=0.5)
        loss_fn.forward(anchor.copy(), positive.copy(), negative.copy())
        grad_a, grad_p, grad_n = loss_fn.backward()

        num_grad_a, num_grad_p, num_grad_n = numerical_gradient_triplet(
            loss_fn, anchor.copy(), positive.copy(), negative.copy()
        )

        assert np.allclose(grad_a, num_grad_a, atol=1e-5)
        assert np.allclose(grad_p, num_grad_p, atol=1e-5)
        assert np.allclose(grad_n, num_grad_n, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_pytorch_comparison(self, random_seed):
        """Compare with PyTorch triplet loss."""
        anchor = np.random.randn(4, 8)
        positive = np.random.randn(4, 8)
        negative = np.random.randn(4, 8)
        margin = 0.5

        # NumPy
        loss_fn = TripletLoss(margin=margin)
        np_loss = loss_fn.forward(anchor.copy(), positive.copy(), negative.copy())
        np_grad_a, np_grad_p, np_grad_n = loss_fn.backward()

        # PyTorch
        anchor_t = torch.tensor(anchor, dtype=torch.float64, requires_grad=True)
        positive_t = torch.tensor(positive, dtype=torch.float64, requires_grad=True)
        negative_t = torch.tensor(negative, dtype=torch.float64, requires_grad=True)

        torch_loss = nn.TripletMarginLoss(margin=margin, p=2)(
            anchor_t, positive_t, negative_t
        )
        torch_loss.backward()

        assert np.isclose(np_loss, torch_loss.item(), atol=1e-5)
        assert np.allclose(np_grad_a, anchor_t.grad.numpy(), atol=1e-5)
        assert np.allclose(np_grad_p, positive_t.grad.numpy(), atol=1e-5)
        assert np.allclose(np_grad_n, negative_t.grad.numpy(), atol=1e-5)


# =============================================================================
# Test Utilities
# =============================================================================


class TestSoftmax:
    """Test softmax utilities."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = np.array([[1.0, 2.0, 3.0]])
        probs = _softmax(x)
        assert np.isclose(probs.sum(), 1.0)

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values."""
        x = np.array([[1000.0, 1001.0, 1002.0]])
        probs = _softmax(x)
        assert not np.any(np.isnan(probs))
        assert np.isclose(probs.sum(), 1.0)

    def test_log_softmax(self):
        """Log-softmax should equal log(softmax)."""
        x = np.array([[1.0, 2.0, 3.0]])
        log_probs = _log_softmax(x)
        probs = _softmax(x)
        expected = np.log(probs)
        assert np.allclose(log_probs, expected)


class TestGetLoss:
    """Test loss factory function."""

    def test_get_mse(self):
        """Get MSE loss by name."""
        loss_fn = get_loss("mse")
        assert isinstance(loss_fn, MSELoss)

    def test_get_cross_entropy(self):
        """Get cross-entropy loss by name."""
        loss_fn = get_loss("cross_entropy")
        assert isinstance(loss_fn, CrossEntropyLoss)

    def test_get_with_kwargs(self):
        """Get loss with keyword arguments."""
        loss_fn = get_loss("focal", gamma=2.0, alpha=0.5)
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.gamma == 2.0
        assert loss_fn.alpha == 0.5

    def test_invalid_name_raises(self):
        """Invalid loss name should raise ValueError."""
        with pytest.raises(ValueError):
            get_loss("invalid_loss")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with MLP."""

    def test_mse_with_mlp(self):
        """Test MSE loss with MLP output."""
        from phase1_basics.mlp import MLP

        np.random.seed(42)
        mlp = MLP(input_size=10, hidden_sizes=[5], output_size=2)
        x = np.random.randn(4, 10)
        target = np.random.randn(4, 2)

        output = mlp.forward(x)
        loss_fn = MSELoss()
        loss = loss_fn.forward(output, target)
        grad = loss_fn.backward()

        mlp.backward(grad)

        # Check gradients were computed
        for param, grad_val in mlp.parameters():
            assert grad_val is not None

    def test_ce_with_mlp(self):
        """Test cross-entropy loss with MLP output."""
        from phase1_basics.mlp import MLP

        np.random.seed(42)
        mlp = MLP(input_size=10, hidden_sizes=[5], output_size=3)
        x = np.random.randn(4, 10)
        target = np.array([0, 1, 2, 0])

        output = mlp.forward(x)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(output, target)
        grad = loss_fn.backward()

        mlp.backward(grad)

        # Check gradients were computed
        for param, grad_val in mlp.parameters():
            assert grad_val is not None


# =============================================================================
# GPU Tests (when PyTorch available)
# =============================================================================

# Lazy check for CUDA availability
def _cuda_available():
    if not HAS_TORCH:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGPU:
    """Test that loss functions work with GPU tensors via PyTorch comparison."""

    @pytest.mark.skipif(
        not _cuda_available(), reason="CUDA not available"
    )
    def test_mse_gpu(self, random_seed):
        """Test MSE works on GPU."""
        pred = np.random.randn(4, 3)
        target = np.random.randn(4, 3)

        # GPU computation via PyTorch
        pred_gpu = torch.tensor(pred, dtype=torch.float64, device="cuda")
        target_gpu = torch.tensor(target, dtype=torch.float64, device="cuda")
        loss_gpu = nn.MSELoss()(pred_gpu, target_gpu)

        # CPU with NumPy
        loss_fn = MSELoss()
        loss_cpu = loss_fn.forward(pred, target)

        assert np.isclose(loss_cpu, loss_gpu.item(), atol=1e-6)

    @pytest.mark.skipif(
        not _cuda_available(), reason="CUDA not available"
    )
    def test_cross_entropy_gpu(self, random_seed):
        """Test cross-entropy works on GPU."""
        logits = np.random.randn(4, 5)
        targets = np.random.randint(0, 5, size=4)

        # GPU computation via PyTorch
        logits_gpu = torch.tensor(logits, dtype=torch.float64, device="cuda")
        targets_gpu = torch.tensor(targets, dtype=torch.long, device="cuda")
        loss_gpu = nn.CrossEntropyLoss()(logits_gpu, targets_gpu)

        # CPU with NumPy
        loss_fn = CrossEntropyLoss()
        loss_cpu = loss_fn.forward(logits, targets)

        assert np.isclose(loss_cpu, loss_gpu.item(), atol=1e-6)

    @pytest.mark.skipif(
        not _cuda_available(), reason="CUDA not available"
    )
    def test_triplet_gpu(self, random_seed):
        """Test triplet loss works on GPU."""
        anchor = np.random.randn(4, 8)
        positive = np.random.randn(4, 8)
        negative = np.random.randn(4, 8)

        # GPU computation via PyTorch
        anchor_gpu = torch.tensor(anchor, dtype=torch.float64, device="cuda")
        positive_gpu = torch.tensor(positive, dtype=torch.float64, device="cuda")
        negative_gpu = torch.tensor(negative, dtype=torch.float64, device="cuda")
        loss_gpu = nn.TripletMarginLoss(margin=0.5)(
            anchor_gpu, positive_gpu, negative_gpu
        )

        # CPU with NumPy
        loss_fn = TripletLoss(margin=0.5)
        loss_cpu = loss_fn.forward(anchor, positive, negative)

        assert np.isclose(loss_cpu, loss_gpu.item(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
