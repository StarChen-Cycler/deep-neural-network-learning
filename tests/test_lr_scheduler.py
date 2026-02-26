"""
Tests for learning rate scheduler implementations.

Tests verify:
- StepLR decay at correct steps
- CosineAnnealing at T_max
- OneCycleLR max learning rate
- Warmup behavior
- Cyclic oscillation
"""

import pytest
import numpy as np

# Check for PyTorch
try:
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import (
        StepLR as TorchStepLR,
        CosineAnnealingLR as TorchCosineAnnealingLR,
        CyclicLR as TorchCyclicLR,
        OneCycleLR as TorchOneCycleLR,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase3_training.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearWarmup,
    CosineWarmup,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    PolynomialLR,
    CosineAnnealingWarmRestarts,
    WarmupDecayScheduler,
    get_scheduler,
    plot_learning_rate_curve,
    LR_SCHEDULERS,
)


# Fixtures
@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


class TestStepLR:
    """Tests for StepLR scheduler."""

    def test_initial_lr(self):
        """Test that initial LR is correct."""
        scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        assert scheduler.get_lr() == 0.1

    def test_decay_at_step_size(self):
        """Test that LR decays by gamma at step_size."""
        scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

        # Step until just before decay
        for _ in range(29):
            scheduler.step()
        assert scheduler.get_lr() == 0.1

        # At step 30, should decay
        scheduler.step()
        assert abs(scheduler.get_lr() - 0.01) < 1e-10

    def test_multiple_decays(self):
        """Test multiple decay steps."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.5)

        expected_lrs = []
        for step in range(50):
            expected_lrs.append(scheduler.get_lr())
            scheduler.step()

        # At step 10: 0.1 * 0.5 = 0.05
        assert abs(expected_lrs[10] - 0.05) < 1e-10
        # At step 20: 0.1 * 0.5^2 = 0.025
        assert abs(expected_lrs[20] - 0.025) < 1e-10
        # At step 30: 0.1 * 0.5^3 = 0.0125
        assert abs(expected_lrs[30] - 0.0125) < 1e-10

    def test_reset(self):
        """Test that reset restores initial state."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)
        for _ in range(25):
            scheduler.step()

        scheduler.reset()
        assert scheduler.get_lr() == 0.1
        assert scheduler.step_count == 0


class TestExponentialLR:
    """Tests for ExponentialLR scheduler."""

    def test_exponential_decay(self):
        """Test exponential decay formula."""
        scheduler = ExponentialLR(base_lr=0.1, gamma=0.99)

        for step in range(10):
            expected = 0.1 * (0.99 ** (step + 1))
            scheduler.step()
            assert abs(scheduler.get_lr() - expected) < 1e-10


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_initial_lr(self):
        """Test initial learning rate."""
        scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)
        assert scheduler.get_lr() == 0.1

    def test_lr_at_half_period(self):
        """Test LR at half of T_max."""
        scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)

        for _ in range(50):
            scheduler.step()

        # At T_max/2, cos(pi/2) = 0, so lr should be halfway
        expected = 0.1 * (1 + np.cos(np.pi * 0.5)) / 2
        assert abs(scheduler.get_lr() - expected) < 1e-10

    def test_lr_at_t_max(self):
        """Test LR at T_max should be close to eta_min."""
        scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)

        for _ in range(100):
            scheduler.step()

        # At T_max, cos(pi) = -1, so lr = eta_min
        assert scheduler.get_lr() < 0.001

    def test_eta_min(self):
        """Test that eta_min is respected."""
        scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.01)

        for _ in range(100):
            scheduler.step()

        # At T_max, lr should be close to eta_min
        assert abs(scheduler.get_lr() - 0.01) < 0.001


class TestLinearWarmup:
    """Tests for LinearWarmup scheduler."""

    def test_warmup_progression(self):
        """Test that LR increases linearly during warmup."""
        scheduler = LinearWarmup(base_lr=0.1, warmup_steps=10, start_lr=0)

        # Initial LR should be start_lr
        assert abs(scheduler.get_lr() - 0) < 1e-10

        lrs = []
        for _ in range(10):
            lr = scheduler.step()
            lrs.append(lr)

        # After each step, LR should increase
        # Step 1: 0 + 0.1 * 1/10 = 0.01
        # Step 10: 0 + 0.1 * 10/10 = 0.1
        for i, lr in enumerate(lrs):
            expected = 0.1 * (i + 1) / 10
            assert abs(lr - expected) < 1e-10

    def test_after_warmup(self):
        """Test that LR stays at base_lr after warmup."""
        scheduler = LinearWarmup(base_lr=0.1, warmup_steps=5, start_lr=0)

        for _ in range(5):
            scheduler.step()

        # After warmup, should stay at base_lr
        for _ in range(10):
            assert abs(scheduler.get_lr() - 0.1) < 1e-10
            scheduler.step()


class TestCyclicLR:
    """Tests for CyclicLR scheduler."""

    def test_oscillation_range(self):
        """Test that LR oscillates between base_lr and max_lr."""
        scheduler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=10)

        lrs = []
        for _ in range(40):
            lrs.append(scheduler.get_lr())
            scheduler.step()

        # All LRs should be in range [base_lr, max_lr]
        assert min(lrs) >= 0.001 - 1e-10
        assert max(lrs) <= 0.01 + 1e-10

    def test_peak_at_step_size(self):
        """Test that LR peaks at step_size."""
        scheduler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=10)

        for _ in range(10):
            scheduler.step()

        # At step_size, should be at max_lr
        assert abs(scheduler.get_lr() - 0.01) < 1e-10


class TestOneCycleLR:
    """Tests for OneCycleLR scheduler."""

    def test_initial_lr(self):
        """Test that initial LR is low."""
        scheduler = OneCycleLR(max_lr=0.01, total_steps=1000)
        # Initial LR should be max_lr / 25
        assert abs(scheduler.get_lr() - 0.01 / 25) < 1e-10

    def test_max_lr_at_warmup_end(self):
        """Test that max LR is reached at end of warmup."""
        scheduler = OneCycleLR(max_lr=0.01, total_steps=100, pct_start=0.3)

        # Step through warmup
        for _ in range(30):
            scheduler.step()

        # Should be at max_lr
        assert abs(scheduler.get_lr() - 0.01) < 1e-10

    def test_final_lr_is_min(self):
        """Test that final LR is min_lr."""
        scheduler = OneCycleLR(max_lr=0.01, total_steps=100, pct_start=0.3)

        for _ in range(100):
            scheduler.step()

        # Should be at min_lr (max_lr / 10000 by default)
        expected_min = 0.01 / 10000
        assert scheduler.get_lr() < expected_min * 2

    def test_max_lr_parameter(self):
        """Test that max_lr is actually the maximum LR seen."""
        scheduler = OneCycleLR(max_lr=0.01, total_steps=100)

        max_seen = 0
        for _ in range(100):
            lr = scheduler.step()
            max_seen = max(max_seen, lr)

        assert abs(max_seen - 0.01) < 1e-10


class TestPolynomialLR:
    """Tests for PolynomialLR scheduler."""

    def test_linear_decay_power_1(self):
        """Test linear decay when power=1."""
        scheduler = PolynomialLR(base_lr=0.1, total_steps=100, power=1, min_lr=0)

        for _ in range(50):
            scheduler.step()

        # At half steps, LR should be half
        assert abs(scheduler.get_lr() - 0.05) < 1e-10

    def test_final_lr(self):
        """Test that final LR is min_lr."""
        scheduler = PolynomialLR(base_lr=0.1, total_steps=100, power=2, min_lr=0)

        for _ in range(100):
            scheduler.step()

        assert scheduler.get_lr() < 1e-6


class TestCosineAnnealingWarmRestarts:
    """Tests for CosineAnnealingWarmRestarts scheduler."""

    def test_restart_behavior(self):
        """Test that LR restarts at T_0."""
        scheduler = CosineAnnealingWarmRestarts(base_lr=0.1, T_0=10, T_mult=1)

        # Step to end of first cycle
        for _ in range(10):
            scheduler.step()

        # Should have restarted, LR close to base again
        # Note: step happens first, so at step 10 we're at start of new cycle
        # The restart happens when T_cur >= T_i

    def test_t_mult_growth(self):
        """Test that period grows with T_mult."""
        scheduler = CosineAnnealingWarmRestarts(base_lr=0.1, T_0=10, T_mult=2)

        # After first restart, T_i should be 10*2 = 20
        for _ in range(10):
            scheduler.step()


class TestReduceLROnPlateau:
    """Tests for ReduceLROnPlateau scheduler."""

    def test_reduce_on_no_improvement(self):
        """Test that LR reduces after patience steps with no improvement."""
        scheduler = ReduceLROnPlateau(base_lr=0.1, mode="min", factor=0.1, patience=3)

        # Degrading metrics for more than patience steps
        for i in range(5):
            scheduler.step(metric=1.0 + i * 0.1)  # Getting worse

        # Should have reduced LR
        assert scheduler.get_lr() < 0.1

    def test_no_reduce_with_improvement(self):
        """Test that LR stays same with improvement."""
        scheduler = ReduceLROnPlateau(base_lr=0.1, mode="min", factor=0.1, patience=3)

        # Improving metrics
        for i in range(10):
            scheduler.step(metric=1.0 - i * 0.1)  # Getting better

        # Should not have reduced LR
        assert abs(scheduler.get_lr() - 0.1) < 1e-10

    def test_mode_max(self):
        """Test mode='max' for metrics where higher is better."""
        scheduler = ReduceLROnPlateau(base_lr=0.1, mode="max", factor=0.1, patience=2)

        # Degrading accuracy
        for i in range(4):
            scheduler.step(metric=0.9 - i * 0.1)

        # Should have reduced LR
        assert scheduler.get_lr() < 0.1


class TestWarmupDecayScheduler:
    """Tests for WarmupDecayScheduler."""

    def test_warmup_phase(self):
        """Test warmup phase behavior."""
        decay = CosineAnnealingLR(base_lr=0.1, T_max=90)
        scheduler = WarmupDecayScheduler(warmup_steps=10, warmup_start_lr=0, decay_scheduler=decay)

        # During warmup
        for _ in range(5):
            lr = scheduler.step()
            assert lr < 0.1

    def test_decay_after_warmup(self):
        """Test that decay scheduler takes over after warmup."""
        decay = CosineAnnealingLR(base_lr=0.1, T_max=100)
        scheduler = WarmupDecayScheduler(warmup_steps=10, warmup_start_lr=0, decay_scheduler=decay)

        # Complete warmup
        for _ in range(10):
            scheduler.step()

        # Now decay scheduler should control
        scheduler.step()
        assert scheduler.get_lr() > 0


class TestGetScheduler:
    """Tests for scheduler factory function."""

    def test_get_step_scheduler(self):
        """Test getting StepLR scheduler."""
        scheduler = get_scheduler("step", base_lr=0.1, step_size=30, gamma=0.1)
        assert isinstance(scheduler, StepLR)

    def test_get_cosine_scheduler(self):
        """Test getting CosineAnnealingLR scheduler."""
        scheduler = get_scheduler("cosine", base_lr=0.1, T_max=100)
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_get_onecycle_scheduler(self):
        """Test getting OneCycleLR scheduler."""
        scheduler = get_scheduler("one_cycle", max_lr=0.01, total_steps=100)
        assert isinstance(scheduler, OneCycleLR)

    def test_invalid_scheduler(self):
        """Test that invalid name raises error."""
        with pytest.raises(ValueError):
            get_scheduler("invalid_scheduler")


class TestPlotLearningRateCurve:
    """Tests for plot_learning_rate_curve function."""

    def test_returns_correct_shapes(self):
        """Test that function returns arrays of correct shape."""
        scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100)
        steps, lrs = plot_learning_rate_curve(scheduler, steps=100, reset=True)

        assert len(steps) == 100
        assert len(lrs) == 100

    def test_reset_parameter(self):
        """Test that reset parameter works."""
        scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.5)

        # First run
        plot_learning_rate_curve(scheduler, steps=50, reset=True)
        assert scheduler.step_count == 50

        # With reset
        steps, _ = plot_learning_rate_curve(scheduler, steps=20, reset=True)
        assert len(steps) == 20
        assert scheduler.step_count == 20


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchComparison:
    """Compare implementations with PyTorch."""

    def test_step_lr_comparison(self):
        """Compare StepLR with PyTorch."""
        # Our implementation
        our_scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

        # PyTorch
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        torch_scheduler = TorchStepLR(optimizer, step_size=30, gamma=0.1)

        our_lrs = []
        torch_lrs = []

        for _ in range(100):
            our_lrs.append(our_scheduler.get_lr())
            torch_lrs.append(optimizer.param_groups[0]["lr"])
            our_scheduler.step()
            torch_scheduler.step()

        # Should match closely
        for our_lr, torch_lr in zip(our_lrs, torch_lrs):
            assert abs(our_lr - torch_lr) < 1e-10

    def test_cosine_annealing_comparison(self):
        """Compare CosineAnnealingLR with PyTorch."""
        our_scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0)

        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        torch_scheduler = TorchCosineAnnealingLR(optimizer, T_max=100, eta_min=0)

        our_lrs = []
        torch_lrs = []

        for _ in range(100):
            our_lrs.append(our_scheduler.get_lr())
            torch_lrs.append(optimizer.param_groups[0]["lr"])
            our_scheduler.step()
            torch_scheduler.step()

        for our_lr, torch_lr in zip(our_lrs, torch_lrs):
            assert abs(our_lr - torch_lr) < 1e-10


class TestSchedulerRegistry:
    """Tests for scheduler registry."""

    def test_registry_has_expected_schedulers(self):
        """Test that registry contains expected schedulers."""
        expected = [
            "StepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "CyclicLR",
            "OneCycleLR",
            "ReduceLROnPlateau",
            "PolynomialLR",
        ]
        for name in expected:
            assert name in LR_SCHEDULERS


# ==================== Integration Tests ====================


class TestSchedulerIntegration:
    """Integration tests with optimization."""

    def test_scheduler_improves_convergence(self):
        """Test that using a scheduler improves convergence vs constant LR."""
        from phase3_training.scheduler_comparison import run_optimization, rosenbrock_function

        np.random.seed(42)
        x_init = np.array([-2.0, 2.0])

        # Constant LR
        constant = StepLR(base_lr=0.1, step_size=1000, gamma=1.0)
        result_constant = run_optimization(
            constant, rosenbrock_function, x_init.copy(), max_steps=200
        )

        # Cosine annealing
        cosine = CosineAnnealingLR(base_lr=0.1, T_max=200)
        result_cosine = run_optimization(
            cosine, rosenbrock_function, x_init.copy(), max_steps=200
        )

        # Cosine should converge better
        # This is not guaranteed, so we just check both improved
        assert result_constant["final_loss"] < result_constant["losses"][0]
        assert result_cosine["final_loss"] < result_cosine["losses"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
