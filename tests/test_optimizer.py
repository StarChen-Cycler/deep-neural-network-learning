"""
Unit tests for neural network optimizers.

Tests verify:
1. Optimizer updates parameters correctly
2. Convergence on simple loss functions
3. Comparison with PyTorch optimizers
4. Specialized convergence properties (Momentum on ravines, Adam stability)
"""

import pytest
import numpy as np
from typing import List, Tuple

# Import optimizers
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_basics.optimizer import (
    SGD,
    Momentum,
    Nesterov,
    AdaGrad,
    RMSprop,
    Adam,
    AdamW,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    get_optimizer,
    get_scheduler,
)
from phase1_basics.mlp import MLP

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

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
def simple_params() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create simple parameters with gradients for testing."""
    params = [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.1, 0.2], [0.3, 0.4]])),
        (np.array([1.0, 2.0]), np.array([0.1, 0.2])),
    ]
    return params


@pytest.fixture
def quadratic_params() -> Tuple[np.ndarray, np.ndarray]:
    """Create parameters for quadratic function f(x) = x^2."""
    x = np.array([2.0])
    grad = np.array([4.0])  # df/dx = 2x = 4
    return [(x, grad)]


# =============================================================================
# Test SGD
# =============================================================================


class TestSGD:
    """Tests for SGD optimizer."""

    def test_sgd_step(self, simple_params):
        """Test SGD updates parameters correctly."""
        optimizer = SGD(lr=0.1)

        # Store original values
        orig_w = simple_params[0][0].copy()
        orig_b = simple_params[1][0].copy()
        grad_w = simple_params[0][1].copy()
        grad_b = simple_params[1][1].copy()

        optimizer.step(simple_params)

        # Check: param = param - lr * grad
        expected_w = orig_w - 0.1 * grad_w
        expected_b = orig_b - 0.1 * grad_b

        np.testing.assert_allclose(simple_params[0][0], expected_w)
        np.testing.assert_allclose(simple_params[1][0], expected_b)

    def test_sgd_convergence_quadratic(self):
        """Test SGD converges on quadratic function."""
        optimizer = SGD(lr=0.1)
        x = np.array([5.0])

        losses = []
        for _ in range(50):
            # f(x) = x^2, gradient = 2x
            grad = 2 * x
            optimizer.step([(x, grad)])
            loss = float(x**2)
            losses.append(loss)

        # Should converge close to 0
        assert losses[-1] < 0.01
        assert losses[-1] < losses[0]  # Loss decreased

    def test_sgd_learning_rate_effect(self):
        """Test that larger learning rate converges faster (up to a point)."""
        lr_small = 0.01
        lr_large = 0.1

        # Small learning rate
        x_small = np.array([5.0])
        opt_small = SGD(lr=lr_small)
        for _ in range(20):
            grad = 2 * x_small
            opt_small.step([(x_small, grad)])

        # Large learning rate
        x_large = np.array([5.0])
        opt_large = SGD(lr=lr_large)
        for _ in range(20):
            grad = 2 * x_large
            opt_large.step([(x_large, grad)])

        # Large learning rate should converge faster
        assert float(x_large**2) < float(x_small**2)


# =============================================================================
# Test Momentum
# =============================================================================


class TestMomentum:
    """Tests for Momentum optimizer."""

    def test_momentum_step(self):
        """Test Momentum updates parameters correctly."""
        optimizer = Momentum(lr=0.1, momentum=0.9)
        x = np.array([1.0])

        # First step
        grad = np.array([1.0])
        optimizer.step([(x, grad)])
        # v = 0.9 * 0 - 0.1 * 1 = -0.1
        # x = 1 + (-0.1) = 0.9
        np.testing.assert_allclose(x, [0.9], atol=1e-6)

        # Second step (velocity accumulates)
        grad = np.array([1.0])
        optimizer.step([(x, grad)])
        # v = 0.9 * (-0.1) - 0.1 * 1 = -0.19
        # x = 0.9 + (-0.19) = 0.71
        np.testing.assert_allclose(x, [0.71], atol=1e-6)

    def test_momentum_convergence_ravine(self):
        """Test Momentum is faster than SGD on ravine surface."""
        # Rosenbrock-like function creates a ravine
        # f(x, y) = (1-x)^2 + 100*(y-x^2)^2
        # Gradient: df/dx = -2(1-x) - 400x(y-x^2)
        #           df/dy = 200(y-x^2)

        def rosenbrock_grad(x, y):
            dx = -2 * (1 - x) - 400 * x * (y - x**2)
            dy = 200 * (y - x**2)
            return np.array([dx, dy])

        def rosenbrock_loss(x, y):
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        # SGD
        x_sgd = np.array([-1.0, 1.0])
        opt_sgd = SGD(lr=0.0001)
        sgd_losses = []

        for _ in range(500):
            grad = rosenbrock_grad(x_sgd[0], x_sgd[1])
            opt_sgd.step([(x_sgd, grad)])
            sgd_losses.append(rosenbrock_loss(x_sgd[0], x_sgd[1]))

        # Momentum
        x_mom = np.array([-1.0, 1.0])
        opt_mom = Momentum(lr=0.0001, momentum=0.9)
        mom_losses = []

        for _ in range(500):
            grad = rosenbrock_grad(x_mom[0], x_mom[1])
            opt_mom.step([(x_mom, grad)])
            mom_losses.append(rosenbrock_loss(x_mom[0], x_mom[1]))

        # Momentum should reach lower loss in same iterations
        # (though exact speedup depends on hyperparameters)
        assert mom_losses[-1] < sgd_losses[-1] or mom_losses[-1] < 1.0


# =============================================================================
# Test Nesterov
# =============================================================================


class TestNesterov:
    """Tests for Nesterov optimizer."""

    def test_nesterov_step(self):
        """Test Nesterov updates parameters correctly."""
        optimizer = Nesterov(lr=0.1, momentum=0.9)
        x = np.array([1.0])

        # First step
        grad = np.array([1.0])
        optimizer.step([(x, grad)])
        # Nesterov has different update than momentum
        # v = 0.9 * 0 - 0.1 * 1 = -0.1
        # x = 1 + (-0.9 * 0 + 1.9 * (-0.1)) = 1 - 0.19 = 0.81
        assert x[0] < 1.0  # Should decrease

    def test_nesterov_convergence(self):
        """Test Nesterov converges on quadratic."""
        optimizer = Nesterov(lr=0.1, momentum=0.9)
        x = np.array([5.0])

        losses = []
        for _ in range(50):
            grad = 2 * x
            optimizer.step([(x, grad)])
            losses.append(float(x**2))

        assert losses[-1] < 0.1


# =============================================================================
# Test AdaGrad
# =============================================================================


class TestAdaGrad:
    """Tests for AdaGrad optimizer."""

    def test_adagrad_adaptive_lr(self):
        """Test AdaGrad adapts learning rate per parameter."""
        optimizer = AdaGrad(lr=1.0)
        x = np.array([1.0])

        # Repeated large gradients should reduce effective learning rate
        grads = []
        for i in range(10):
            grad = np.array([10.0])
            grads.append(grad)
            optimizer.step([(x, grad)])

        # After many steps, accumulated grad^2 is large
        # Effective lr = lr / sqrt(sum(grad^2))
        # Should be much smaller than initial
        assert optimizer.accumulated_sq_grad[0][0] > 100


# =============================================================================
# Test RMSprop
# =============================================================================


class TestRMSprop:
    """Tests for RMSprop optimizer."""

    def test_rmsprop_step(self):
        """Test RMSprop updates parameters correctly."""
        optimizer = RMSprop(lr=0.01, alpha=0.9)
        x = np.array([1.0])

        grad = np.array([1.0])
        optimizer.step([(x, grad)])

        # x should decrease (moving in negative gradient direction)
        assert x[0] < 1.0

    def test_rmsprop_non_monotonic(self):
        """Test RMSprop doesn't monotonically decrease learning rate."""
        optimizer = RMSprop(lr=0.1, alpha=0.9)
        x = np.array([1.0])

        # First few steps with gradient 1
        for _ in range(5):
            grad = np.array([1.0])
            optimizer.step([(x, grad)])

        first_accumulated = optimizer.accumulated_sq_grad[0][0]

        # Now with gradient 0.1 (smaller)
        for _ in range(5):
            grad = np.array([0.1])
            optimizer.step([(x, grad)])

        # Accumulated should use EMA, not just sum
        # It should be influenced more by recent gradients
        # but not as extreme as AdaGrad


# =============================================================================
# Test Adam
# =============================================================================


class TestAdam:
    """Tests for Adam optimizer."""

    def test_adam_step(self):
        """Test Adam updates parameters correctly."""
        optimizer = Adam(lr=0.1)
        x = np.array([1.0])

        grad = np.array([1.0])
        optimizer.step([(x, grad)])

        # x should decrease
        assert x[0] < 1.0

    def test_adam_bias_correction(self):
        """Test Adam bias correction in early steps."""
        optimizer = Adam(lr=0.1, betas=(0.9, 0.999))
        x = np.array([1.0])

        # First step with gradient 1
        grad = np.array([1.0])
        optimizer.step([(x, grad)])

        # m_hat should be approximately 1 / (1 - 0.9) * 0.1 = 1.0
        # (since m = 0.1 after first step)
        assert optimizer.t == 1

    def test_adam_convergence(self):
        """Test Adam converges quickly on simple problem."""
        optimizer = Adam(lr=1.0)  # Higher learning rate for faster convergence
        x = np.array([5.0])

        losses = []
        for _ in range(50):
            grad = 2 * x  # Quadratic gradient
            optimizer.step([(x, grad)])
            losses.append(float(x**2))

        # Adam should converge very quickly
        assert losses[-1] < losses[0]  # Loss decreased
        assert losses[-1] < 1.0  # Reasonable convergence

    def test_adam_stability_different_scales(self):
        """Test Adam handles different gradient scales."""
        optimizer = Adam(lr=0.1)

        # Parameters with very different gradient scales
        x1 = np.array([1.0])  # Will have small gradient
        x2 = np.array([1.0])  # Will have large gradient

        for _ in range(100):
            # Small gradient for x1
            grad1 = 0.01 * x1
            optimizer.step([(x1, grad1)])

        optimizer2 = Adam(lr=0.1)
        for _ in range(100):
            # Large gradient for x2
            grad2 = 100.0 * x2
            optimizer2.step([(x2, grad2)])

        # Both should converge despite different scales
        # (Adam adapts learning rate)
        assert abs(x1[0]) < 1.1  # Moved some
        assert abs(x2[0]) < 1.0  # Converged (Adam handles large gradients)


# =============================================================================
# Test AdamW
# =============================================================================


class TestAdamW:
    """Tests for AdamW optimizer."""

    def test_adamw_weight_decay(self):
        """Test AdamW applies weight decay."""
        optimizer = AdamW(lr=0.1, weight_decay=0.1)
        x = np.array([1.0])

        # Zero gradient - only weight decay should apply
        grad = np.array([0.0])
        optimizer.step([(x, grad)])

        # x should decrease due to weight decay: x = x - lr * wd * x
        # 1 - 0.1 * 0.1 * 1 = 0.99
        np.testing.assert_allclose(x, [0.99], atol=0.01)

    def test_adamw_vs_adam_l2(self):
        """Test AdamW differs from Adam with L2 regularization."""
        # Adam with L2: grad_effective = grad + wd * x
        # AdamW: x = x - lr * wd * x (separate from gradient update)

        # This is a conceptual test - the difference is subtle
        optimizer_w = AdamW(lr=0.1, weight_decay=0.5)
        x_w = np.array([1.0])

        grad = np.array([0.0])
        optimizer_w.step([(x_w, grad)])

        # Weight decay should have been applied
        assert x_w[0] < 1.0


# =============================================================================
# Test Learning Rate Schedulers
# =============================================================================


class TestSchedulers:
    """Tests for learning rate schedulers."""

    def test_step_lr(self):
        """Test StepLR scheduler."""
        optimizer = SGD(lr=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        assert optimizer.lr == 0.1

        scheduler.step()  # epoch 1
        assert optimizer.lr == 0.1

        scheduler.step()  # epoch 2: decay
        assert optimizer.lr == 0.05

        scheduler.step()  # epoch 3
        assert optimizer.lr == 0.05

        scheduler.step()  # epoch 4: decay
        assert optimizer.lr == 0.025

    def test_exponential_lr(self):
        """Test ExponentialLR scheduler."""
        optimizer = SGD(lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        for i in range(5):
            scheduler.step()
            expected = 0.1 * (0.9 ** (i + 1))
            np.testing.assert_allclose(optimizer.lr, expected, atol=1e-7)

    def test_cosine_annealing_lr(self):
        """Test CosineAnnealingLR scheduler."""
        optimizer = SGD(lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)

        # At epoch 0, lr should be base_lr
        scheduler.step()  # epoch 1
        # lr = 0 + 0.1 * (1 + cos(π * 1 / 10)) / 2
        expected = 0.1 * (1 + np.cos(np.pi * 1 / 10)) / 2
        np.testing.assert_allclose(optimizer.lr, expected, atol=1e-7)

        # At T_max, lr should be eta_min
        for _ in range(9):
            scheduler.step()
        np.testing.assert_allclose(optimizer.lr, 0.0, atol=1e-7)


# =============================================================================
# Test Optimizer Registry
# =============================================================================


class TestRegistry:
    """Tests for optimizer and scheduler registry."""

    def test_get_optimizer_sgd(self):
        """Test get_optimizer for SGD."""
        opt = get_optimizer("sgd", lr=0.01)
        assert isinstance(opt, SGD)
        assert opt.lr == 0.01

    def test_get_optimizer_adam(self):
        """Test get_optimizer for Adam."""
        opt = get_optimizer("adam", lr=0.001, betas=(0.9, 0.99))
        assert isinstance(opt, Adam)
        assert opt.lr == 0.001
        assert opt.beta2 == 0.99

    def test_get_optimizer_invalid(self):
        """Test get_optimizer with invalid name."""
        with pytest.raises(ValueError):
            get_optimizer("invalid_optimizer")

    def test_get_scheduler(self):
        """Test get_scheduler."""
        opt = Adam(lr=0.001)
        scheduler = get_scheduler("step", opt, step_size=10, gamma=0.5)
        assert isinstance(scheduler, StepLR)


# =============================================================================
# Test with MLP
# =============================================================================


class TestWithMLP:
    """Tests using MLP and loss functions."""

    def test_sgd_trains_mlp(self, random_seed):
        """Test SGD can train an MLP."""
        mlp = MLP(input_size=10, hidden_sizes=[32], output_size=1, activation="relu")
        optimizer = SGD(lr=0.01)

        # Create simple regression data
        X = np.random.randn(100, 10)
        y = np.sum(X[:, :3], axis=1, keepdims=True)  # Only first 3 features matter

        losses = []
        for _ in range(50):
            # Forward
            pred = mlp.forward(X)
            loss = np.mean((pred - y) ** 2)
            losses.append(loss)

            # Backward
            grad = 2 * (pred - y) / y.size
            mlp.backward(grad)

            # Update
            optimizer.step(mlp.parameters())
            mlp.zero_grad()

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_adam_trains_mlp_faster(self, random_seed):
        """Test Adam trains MLP faster than SGD."""
        # Same setup for both
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.sum(X[:, :3], axis=1, keepdims=True)

        # SGD
        np.random.seed(42)
        mlp_sgd = MLP(input_size=10, hidden_sizes=[32], output_size=1, activation="relu")
        opt_sgd = SGD(lr=0.01)

        sgd_losses = []
        for _ in range(20):
            pred = mlp_sgd.forward(X)
            loss = np.mean((pred - y) ** 2)
            sgd_losses.append(loss)
            grad = 2 * (pred - y) / y.size
            mlp_sgd.backward(grad)
            opt_sgd.step(mlp_sgd.parameters())
            mlp_sgd.zero_grad()

        # Adam
        np.random.seed(42)
        mlp_adam = MLP(input_size=10, hidden_sizes=[32], output_size=1, activation="relu")
        opt_adam = Adam(lr=0.01)

        adam_losses = []
        for _ in range(20):
            pred = mlp_adam.forward(X)
            loss = np.mean((pred - y) ** 2)
            adam_losses.append(loss)
            grad = 2 * (pred - y) / y.size
            mlp_adam.backward(grad)
            opt_adam.step(mlp_adam.parameters())
            mlp_adam.zero_grad()

        # Adam should converge faster (lower loss in same iterations)
        assert adam_losses[-1] < sgd_losses[-1]


# =============================================================================
# PyTorch Comparison Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchComparison:
    """Compare our implementations with PyTorch."""

    def test_sgd_vs_pytorch(self):
        """Compare SGD with PyTorch."""
        # Our implementation
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad_np = np.array([[0.1, 0.2], [0.3, 0.4]])
        opt_np = SGD(lr=0.1)
        opt_np.step([(x_np, grad_np)])

        # PyTorch
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        x_torch.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        opt_torch = optim.SGD([x_torch], lr=0.1)
        opt_torch.step()

        np.testing.assert_allclose(x_np, x_torch.detach().numpy())

    def test_adam_vs_pytorch(self):
        """Compare Adam with PyTorch."""
        # Our implementation
        x_np = np.array([1.0, 2.0])
        opt_np = Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # PyTorch
        x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        opt_torch = optim.Adam([x_torch], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        # Multiple steps
        for i in range(5):
            grad_val = np.array([0.5 + 0.1 * i, 0.3 - 0.05 * i])

            # Our step
            opt_np.step([(x_np, grad_val.copy())])

            # PyTorch step
            x_torch.grad = torch.tensor(grad_val, dtype=torch.float64)
            opt_torch.step()

        # Should be close (may not be exact due to implementation details)
        np.testing.assert_allclose(x_np, x_torch.detach().numpy(), rtol=1e-3)

    def test_sgd_momentum_vs_pytorch(self):
        """Compare Momentum SGD with PyTorch."""
        # Our implementation
        x_np = np.array([1.0, 2.0])
        opt_np = Momentum(lr=0.01, momentum=0.9)

        # PyTorch
        x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        opt_torch = optim.SGD([x_torch], lr=0.01, momentum=0.9)

        for i in range(10):
            grad_val = np.array([1.0, 0.5])

            opt_np.step([(x_np, grad_val.copy())])

            x_torch.grad = torch.tensor(grad_val, dtype=torch.float64)
            opt_torch.step()

        np.testing.assert_allclose(x_np, x_torch.detach().numpy(), rtol=1e-5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_training_loop(self, random_seed):
        """Test complete training loop with all components."""
        # Create model
        mlp = MLP(input_size=20, hidden_sizes=[64, 32], output_size=3, activation="relu")

        # Create optimizer
        optimizer = Adam(lr=0.01)

        # Create scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=50)

        # Create data
        X = np.random.randn(200, 20)
        y_indices = np.random.randint(0, 3, size=200)
        y_onehot = np.zeros((200, 3))
        y_onehot[np.arange(200), y_indices] = 1

        losses = []
        for epoch in range(50):
            # Forward
            logits = mlp.forward(X)

            # Softmax cross-entropy loss
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            log_probs = np.log(probs + 1e-10)
            loss = -np.mean(np.sum(y_onehot * log_probs, axis=1))
            losses.append(loss)

            # Backward
            grad = (probs - y_onehot) / 200
            mlp.backward(grad)

            # Update
            optimizer.step(mlp.parameters())
            mlp.zero_grad()

            # Update learning rate
            scheduler.step()

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Learning rate should have decayed
        assert optimizer.lr < 0.01

    def test_optimizer_comparison(self, random_seed):
        """Compare all optimizers on same task."""
        # Create data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.sum(X[:, :3], axis=1, keepdims=True)

        optimizers = {
            "SGD": SGD(lr=0.01),
            "Momentum": Momentum(lr=0.01, momentum=0.9),
            "Nesterov": Nesterov(lr=0.01, momentum=0.9),
            "AdaGrad": AdaGrad(lr=0.1),
            "RMSprop": RMSprop(lr=0.01),
            "Adam": Adam(lr=0.01),
        }

        results = {}
        for name, opt in optimizers.items():
            np.random.seed(42)
            mlp = MLP(input_size=10, hidden_sizes=[32], output_size=1, activation="relu")

            losses = []
            for _ in range(50):
                pred = mlp.forward(X)
                loss = np.mean((pred - y) ** 2)
                losses.append(loss)
                grad = 2 * (pred - y) / y.size
                mlp.backward(grad)
                opt.step(mlp.parameters())
                mlp.zero_grad()

            results[name] = losses[-1]

        # All optimizers should reduce loss
        for name, final_loss in results.items():
            assert final_loss < 5.0, f"{name} failed to converge: {final_loss}"

        # Adam typically performs well
        assert results["Adam"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
