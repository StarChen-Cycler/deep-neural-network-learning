"""
Tests for NaN debugger and training stability module.

Tests cover:
    - Data validation (NaN/Inf detection)
    - NaN debugger detection and diagnosis
    - Training stability monitor (gradient norm warnings)
    - Auto recovery handler (LR reduction)
    - Numerical stability tests
"""

import pytest
import numpy as np
from typing import List

from phase4_advanced.nan_debugger import (
    # Components
    NaNDebugger,
    TrainingStabilityMonitor,
    AutoRecoveryHandler,
    DataValidator,
    NumericalStabilityTester,
    # Enums
    StabilityStatus,
    RecoveryAction,
    # Data classes
    DiagnosticResult,
    StabilityReport,
    # Utilities
    safe_log,
    safe_exp,
    safe_divide,
    detect_anomaly,
    get_nan_debugger,
)

# Try to import PyTorch for comparison
try:
    import torch

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
        np.random.randn(32, 64) * 0.1,
        np.random.randn(64, 128) * 0.1,
        np.random.randn(128, 10) * 0.1,
    ]


@pytest.fixture
def large_gradients() -> List[np.ndarray]:
    """Create large gradients for testing warnings."""
    return [
        np.random.randn(32, 64) * 100,
        np.random.randn(64, 128) * 100,
    ]


# =============================================================================
# Test DataValidator
# =============================================================================


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_valid_data_passes(self):
        """Test that valid data passes validation."""
        validator = DataValidator()
        X = np.random.randn(100, 10)
        result = validator.validate(X)

        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_nan_detection(self):
        """Test NaN detection in data."""
        validator = DataValidator()
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan

        result = validator.validate(X)

        assert result["is_valid"] is False
        assert any("NaN" in issue for issue in result["issues"])
        assert result["stats"]["nan_count"] == 1

    def test_inf_detection(self):
        """Test Inf detection in data."""
        validator = DataValidator()
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf

        result = validator.validate(X)

        assert result["is_valid"] is False
        assert any("Inf" in issue for issue in result["issues"])
        assert result["stats"]["inf_count"] == 2

    def test_extreme_value_warning(self):
        """Test extreme value warning."""
        validator = DataValidator(max_value=100)
        X = np.random.randn(100, 10)
        X[0, 0] = 1000  # Extreme value

        result = validator.validate(X)

        assert any("Extreme" in w for w in result["warnings"])

    def test_label_validation(self):
        """Test label validation."""
        validator = DataValidator()
        X = np.random.randn(100, 10)
        y = np.array([0, 1, np.nan, 1, 0] * 20)

        result = validator.validate(X, y)

        assert any("NaN" in issue for issue in result["label_issues"])

    def test_clean_fill_zero(self):
        """Test data cleaning with fill_zero strategy."""
        validator = DataValidator()
        X = np.array([[1.0, np.nan], [np.inf, 2.0]])
        cleaned = validator.clean(X, strategy="fill_zero")

        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))

    def test_clean_fill_mean(self):
        """Test data cleaning with fill_mean strategy."""
        validator = DataValidator()
        X = np.array([[1.0, 2.0], [np.nan, 4.0]])
        cleaned = validator.clean(X, strategy="fill_mean")

        assert not np.any(np.isnan(cleaned))
        # NaN should be replaced with mean of valid values [1.0, 2.0, 4.0] = 2.333...
        assert np.isclose(cleaned[1, 0], np.mean([1.0, 2.0, 4.0]))

    def test_numerical_stability_check(self):
        """Test numerical stability warnings."""
        validator = DataValidator(check_numerical_stability=True)
        X = np.random.randn(100, 10)
        X[0, 0] = 800  # Would cause exp overflow

        result = validator.validate(X)

        assert any("overflow" in s for s in result["stability_issues"])


# =============================================================================
# Test NaNDebugger
# =============================================================================


class TestNaNDebugger:
    """Tests for NaNDebugger class."""

    def test_healthy_loss(self):
        """Test that healthy loss is not flagged."""
        debugger = NaNDebugger()

        is_nan = debugger.check_loss(0.5)

        assert is_nan is False

    def test_nan_loss_detection(self):
        """Test NaN loss detection."""
        debugger = NaNDebugger()

        is_nan = debugger.check_loss(float("nan"))

        assert is_nan is True

    def test_inf_loss_detection(self):
        """Test Inf loss detection."""
        debugger = NaNDebugger()

        is_nan = debugger.check_loss(float("inf"))

        assert is_nan is True

    def test_abnormal_loss_detection(self):
        """Test abnormal loss detection."""
        debugger = NaNDebugger(loss_threshold=1e6)

        is_nan = debugger.check_loss(1e10)

        assert is_nan is True

    def test_gradient_nan_detection(self):
        """Test gradient NaN detection."""
        debugger = NaNDebugger()
        grads = [np.array([1.0, 2.0]), np.array([np.nan, 1.0])]

        has_problem, norm = debugger.check_gradients(grads)

        assert has_problem is True

    def test_gradient_norm_warning(self):
        """Test gradient norm warning threshold."""
        debugger = NaNDebugger(gradient_threshold=100.0)
        grads = [np.random.randn(100, 100) * 20]  # Large gradients

        has_problem, norm = debugger.check_gradients(grads)

        # Should trigger if norm > 100
        if norm > 100:
            assert has_problem == True  # Use == for numpy.bool_ comparison

    def test_diagnose_healthy(self, sample_gradients):
        """Test diagnosis with healthy values."""
        debugger = NaNDebugger()

        report = debugger.diagnose(
            gradients=sample_gradients, loss=0.5, learning_rate=0.001
        )

        assert report.status == StabilityStatus.HEALTHY
        assert len(report.recommendations) == 0

    def test_diagnose_nan_loss(self):
        """Test diagnosis with NaN loss."""
        debugger = NaNDebugger()

        report = debugger.diagnose(loss=float("nan"))

        assert report.status == StabilityStatus.CRITICAL
        assert report.recovery_action == RecoveryAction.REDUCE_LR
        assert any("NaN" in r for r in report.recommendations)

    def test_diagnose_high_lr(self):
        """Test diagnosis with high learning rate."""
        debugger = NaNDebugger()

        report = debugger.diagnose(loss=0.5, learning_rate=0.5)

        assert any("learning rate" in r.lower() for r in report.recommendations)

    def test_loss_history(self):
        """Test loss history tracking."""
        debugger = NaNDebugger(history_size=10)

        for i in range(15):
            debugger.record_loss(float(i))

        history = debugger.get_history()

        assert len(history["loss"]) == 10
        assert history["loss"][-1] == 14.0

    def test_reset(self):
        """Test debugger reset."""
        debugger = NaNDebugger()
        debugger.check_loss(float("nan"))
        debugger.record_loss(1.0)

        debugger.reset()

        assert len(debugger.get_history()["loss"]) == 0


# =============================================================================
# Test TrainingStabilityMonitor
# =============================================================================


class TestTrainingStabilityMonitor:
    """Tests for TrainingStabilityMonitor class.

    SUCCESS CRITERION: Gradient norm > 100 triggers warning
    """

    def test_gradient_norm_warning_triggers(self, large_gradients):
        """Test that gradient norm > 100 triggers warning.

        SUCCESS CRITERION: 梯度范数>100时触发警告
        """
        monitor = TrainingStabilityMonitor(
            grad_norm_threshold=100.0, enable_warnings=False
        )

        # Create gradients with norm > 100
        grads = [np.random.randn(100, 100) * 50]  # Should have norm >> 100

        report = monitor.check(gradients=grads, loss=0.5)

        # Check if gradient norm exceeds threshold
        _, grad_norm = monitor.debugger.check_gradients(grads)

        if grad_norm > 100:
            assert report.status == StabilityStatus.WARNING
            assert any(
                "100" in d.message for d in report.diagnostics if not d.passed
            )

    def test_healthy_training(self, sample_gradients):
        """Test monitoring with healthy training."""
        monitor = TrainingStabilityMonitor(enable_warnings=False)

        report = monitor.check(gradients=sample_gradients, loss=0.5)

        assert report.status == StabilityStatus.HEALTHY

    def test_loss_spike_detection(self):
        """Test loss spike detection."""
        monitor = TrainingStabilityMonitor(
            loss_spike_threshold=5.0, enable_warnings=False
        )

        # Build up some history
        for _ in range(10):
            monitor.check(loss=1.0)

        # Spike the loss
        report = monitor.check(loss=10.0)  # 10x spike

        # Should detect spike
        assert any(
            "spike" in d.name for d in report.diagnostics if not d.passed
        ) or report.status == StabilityStatus.WARNING

    def test_nan_loss_critical(self):
        """Test that NaN loss is critical."""
        monitor = TrainingStabilityMonitor(enable_warnings=False)

        report = monitor.check(loss=float("nan"))

        assert report.status == StabilityStatus.CRITICAL

    def test_get_stats(self):
        """Test statistics retrieval."""
        monitor = TrainingStabilityMonitor(enable_warnings=False)

        for i in range(10):
            monitor.check(loss=float(i))

        stats = monitor.get_stats()

        assert stats["total_steps"] == 10
        assert "loss_mean" in stats
        assert np.isclose(stats["loss_mean"], 4.5)

    def test_warning_count(self, large_gradients):
        """Test warning count increment."""
        monitor = TrainingStabilityMonitor(
            grad_norm_threshold=1.0, enable_warnings=False
        )  # Low threshold

        # Create large gradients
        grads = [np.random.randn(100, 100) * 10]

        for _ in range(3):
            monitor.check(gradients=grads)

        stats = monitor.get_stats()

        assert stats["warning_count"] >= 1


# =============================================================================
# Test AutoRecoveryHandler
# =============================================================================


class TestAutoRecoveryHandler:
    """Tests for AutoRecoveryHandler class.

    SUCCESS CRITERION: NaN detection triggers automatic LR reduction, training recovers
    """

    def test_lr_reduction_on_nan(self):
        """Test that NaN triggers LR reduction.

        SUCCESS CRITERION: NaN检测触发后自动降低学习率,训练恢复
        """
        handler = AutoRecoveryHandler(
            initial_lr=0.001,
            lr_reduction_factor=0.1,
            enable_auto_clip=False,
        )

        # Trigger with NaN loss
        recovered = handler.check_instability(loss=float("nan"))

        assert recovered is True
        assert handler.get_current_lr() == 0.0001  # 0.001 * 0.1

    def test_multiple_reductions(self):
        """Test multiple LR reductions."""
        handler = AutoRecoveryHandler(
            initial_lr=0.001,
            lr_reduction_factor=0.1,
            max_reduction_count=3,
        )

        # Trigger multiple reductions
        for _ in range(3):
            handler.check_instability(loss=float("nan"))

        assert handler._reduction_count == 3

        # Should not reduce further
        old_lr = handler.get_current_lr()
        handler.check_instability(loss=float("nan"))
        assert handler.get_current_lr() == old_lr

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        handler = AutoRecoveryHandler(
            initial_lr=0.001,
            enable_auto_clip=True,
            grad_clip_max_norm=1.0,
        )

        grads = [np.random.randn(100, 100) * 10]  # Large gradients
        clipped, original_norm = handler.clip_gradients(grads)

        # Check original norm was large
        assert original_norm > 1.0

        # Check clipped norm is within limit
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped))
        assert clipped_norm <= 1.0 + 1e-6

    def test_min_lr_limit(self):
        """Test minimum LR limit."""
        handler = AutoRecoveryHandler(
            initial_lr=0.001,
            lr_reduction_factor=0.1,
            min_lr=1e-6,
            max_reduction_count=10,
        )

        # Reduce many times
        for _ in range(10):
            handler.check_instability(loss=float("nan"))

        assert handler.get_current_lr() >= 1e-6

    def test_reset_lr(self):
        """Test LR reset."""
        handler = AutoRecoveryHandler(initial_lr=0.001)

        handler.check_instability(loss=float("nan"))
        handler.reset_lr()

        assert handler.get_current_lr() == 0.001

    def test_can_continue(self):
        """Test continuation check."""
        handler = AutoRecoveryHandler(
            initial_lr=0.001, max_reduction_count=2
        )

        assert handler.can_continue() is True

        for _ in range(2):
            handler.check_instability(loss=float("nan"))

        assert handler.can_continue() is False

    def test_recovery_history(self):
        """Test recovery history tracking."""
        handler = AutoRecoveryHandler(initial_lr=0.001)

        handler.check_instability(loss=float("nan"))

        history = handler.get_recovery_history()

        assert len(history) == 1
        assert history[0]["type"] == "lr_reduction"


# =============================================================================
# Test NumericalStabilityTester
# =============================================================================


class TestNumericalStabilityTester:
    """Tests for NumericalStabilityTester class.

    SUCCESS CRITERION: Stability tests pass on all data types
    """

    def test_exp_stability(self):
        """Test exp stability across data types.

        SUCCESS CRITERION: 数值稳定性测试在所有数据类型上通过
        """
        tester = NumericalStabilityTester()
        passed, message = tester.test_exp_stability()

        assert passed is True, message

    def test_log_stability(self):
        """Test log stability across data types.

        SUCCESS CRITERION: 数值稳定性测试在所有数据类型上通过
        """
        tester = NumericalStabilityTester()
        passed, message = tester.test_log_stability()

        assert passed is True, message

    def test_softmax_stability(self):
        """Test softmax stability across data types.

        SUCCESS CRITERION: 数值稳定性测试在所有数据类型上通过
        """
        tester = NumericalStabilityTester()
        passed, message = tester.test_softmax_stability()

        assert passed is True, message

    def test_division_stability(self):
        """Test division stability across data types.

        SUCCESS CRITERION: 数值稳定性测试在所有数据类型上通过
        """
        tester = NumericalStabilityTester()
        passed, message = tester.test_division_stability()

        assert passed is True, message

    def test_run_all_tests(self):
        """Test running all stability tests.

        SUCCESS CRITERION: 数值稳定性测试在所有数据类型上通过
        """
        tester = NumericalStabilityTester()
        results = tester.run_all_tests()

        assert results["all_passed"] is True
        assert "exp_stability" in results
        assert "log_stability" in results
        assert "softmax_stability" in results
        assert "division_stability" in results

    def test_float32_stability(self):
        """Test float32 specific stability."""
        tester = NumericalStabilityTester()
        results = tester.run_all_tests()

        assert results["details"].get("exp_float32", False) is True
        assert results["details"].get("log_float32", False) is True
        assert results["details"].get("softmax_float32", False) is True
        assert results["details"].get("division_float32", False) is True

    def test_float64_stability(self):
        """Test float64 specific stability."""
        tester = NumericalStabilityTester()
        results = tester.run_all_tests()

        assert results["details"].get("exp_float64", False) is True
        assert results["details"].get("log_float64", False) is True
        assert results["details"].get("softmax_float64", False) is True
        assert results["details"].get("division_float64", False) is True


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_safe_log(self):
        """Test safe log function."""
        x = np.array([0.0, 1e-100, 1.0, 10.0])

        result = safe_log(x)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_safe_exp(self):
        """Test safe exp function."""
        x = np.array([0.0, 100, 1000, -1000])

        result = safe_exp(x)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_safe_divide(self):
        """Test safe divide function."""
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([1.0, 0.0, 0.5])

        result = safe_divide(numerator, denominator)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_detect_anomaly_nan(self):
        """Test anomaly detection for NaN."""
        x = np.array([1.0, np.nan, 3.0])

        anomalies = detect_anomaly(x, "test_tensor")

        assert len(anomalies) == 1
        assert "NaN" in anomalies[0]

    def test_detect_anomaly_inf(self):
        """Test anomaly detection for Inf."""
        x = np.array([1.0, np.inf, -np.inf])

        anomalies = detect_anomaly(x, "test_tensor")

        assert len(anomalies) == 1
        assert "Inf" in anomalies[0]

    def test_detect_anomaly_healthy(self):
        """Test anomaly detection for healthy tensor."""
        x = np.array([1.0, 2.0, 3.0])

        anomalies = detect_anomaly(x, "test_tensor")

        assert len(anomalies) == 0


# =============================================================================
# Test Registry
# =============================================================================


class TestRegistry:
    """Tests for component registry."""

    def test_get_nan_debugger(self):
        """Test getting NaNDebugger from registry."""
        cls = get_nan_debugger("NaNDebugger")
        assert cls == NaNDebugger

    def test_get_monitor(self):
        """Test getting TrainingStabilityMonitor from registry."""
        cls = get_nan_debugger("TrainingStabilityMonitor")
        assert cls == TrainingStabilityMonitor

    def test_get_recovery_handler(self):
        """Test getting AutoRecoveryHandler from registry."""
        cls = get_nan_debugger("AutoRecoveryHandler")
        assert cls == AutoRecoveryHandler

    def test_get_validator(self):
        """Test getting DataValidator from registry."""
        cls = get_nan_debugger("DataValidator")
        assert cls == DataValidator

    def test_get_invalid(self):
        """Test getting invalid component raises error."""
        with pytest.raises(ValueError):
            get_nan_debugger("InvalidComponent")


# =============================================================================
# Test Integration
# =============================================================================


class TestIntegration:
    """Integration tests for NaN debugging workflow."""

    def test_full_debugging_workflow(self):
        """Test complete debugging workflow."""
        # Setup
        validator = DataValidator()
        debugger = NaNDebugger()
        monitor = TrainingStabilityMonitor(grad_norm_threshold=100.0)
        handler = AutoRecoveryHandler(initial_lr=0.001)

        # Validate data
        X = np.random.randn(100, 10)
        data_report = validator.validate(X)
        assert data_report["is_valid"]

        # Simulate training steps
        for step in range(10):
            loss = 1.0 / (step + 1)  # Decreasing loss
            grads = [np.random.randn(32, 64) * 0.1]

            # Check stability
            report = monitor.check(gradients=grads, loss=loss)

            # Auto recovery
            if handler.check_instability(loss, grads):
                new_lr = handler.get_current_lr()
                assert new_lr <= 0.001

        assert monitor.get_stats()["total_steps"] == 10

    def test_nan_recovery_scenario(self):
        """Test recovery from NaN scenario.

        SUCCESS CRITERION: NaN检测触发后自动降低学习率,训练恢复
        """
        handler = AutoRecoveryHandler(
            initial_lr=0.01,
            lr_reduction_factor=0.1,
        )

        # Simulate training with NaN occurrence
        losses = [1.0, 0.5, float("nan"), 0.3, 0.2]

        current_lr = handler.get_current_lr()
        nan_handled = False

        for loss in losses:
            if np.isnan(loss):
                handler.check_instability(loss=loss)
                new_lr = handler.get_current_lr()
                if new_lr < current_lr:
                    nan_handled = True
                    current_lr = new_lr
            else:
                handler.check_instability(loss=loss)

        assert nan_handled is True
        assert handler.get_current_lr() < 0.01

    def test_data_validation_cleaning_workflow(self):
        """Test data validation and cleaning workflow."""
        validator = DataValidator()

        # Create dirty data
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        X[2, 2] = -np.inf

        # Validate
        report = validator.validate(X)
        assert not report["is_valid"]

        # Clean
        X_clean = validator.clean(X, strategy="fill_mean")

        # Re-validate
        report = validator.validate(X_clean)
        assert report["is_valid"]


# =============================================================================
# PyTorch Comparison Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPyTorchComparison:
    """Tests comparing with PyTorch implementations."""

    def test_gradient_clipping_matches_pytorch(self):
        """Test that our gradient clipping matches PyTorch."""
        from phase4_advanced.gradient_stability import clip_grad_norm

        # Create sample gradients with fixed seed for reproducibility
        np.random.seed(42)
        np_grads = [np.random.randn(32, 64) * 10, np.random.randn(64, 10) * 10]

        # Our implementation
        clipped_np, norm_np = clip_grad_norm(np_grads, max_norm=1.0)

        # PyTorch implementation - need requires_grad for clip_grad_norm_
        torch_params = [torch.tensor(g.copy(), dtype=torch.float64, requires_grad=True) for g in np_grads]
        # Simulate gradients by setting .grad attribute
        for p, g in zip(torch_params, np_grads):
            p.grad = torch.tensor(g.copy(), dtype=torch.float64)

        norm_torch = torch.nn.utils.clip_grad_norm_(torch_params, max_norm=1.0)

        assert np.isclose(norm_np, norm_torch.item(), rtol=1e-5)

        # Check clipped gradients match
        for np_g, torch_p in zip(clipped_np, torch_params):
            assert np.allclose(np_g, torch_p.grad.numpy(), rtol=1e-5)

    def test_nan_detection_matches_pytorch(self):
        """Test NaN detection matches PyTorch behavior."""
        # Our implementation
        debugger = NaNDebugger()
        detected = debugger.check_loss(float("nan"))

        # PyTorch behavior
        torch_loss = torch.tensor(float("nan"))
        torch_detected = torch.isnan(torch_loss).item()

        assert detected == torch_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
