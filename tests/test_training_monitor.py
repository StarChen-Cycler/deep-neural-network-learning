"""
Tests for Training Monitor module.

Tests cover:
    - TrainingMonitor core functionality
    - GradientReport, ActivationReport, WeightUpdateReport
    - Health checking and warnings
    - Dead neuron detection
    - Activation distribution computation
"""

import pytest
import numpy as np

from phase4_advanced.training_monitor import (
    TrainingMonitor,
    GradientReport,
    ActivationReport,
    WeightUpdateReport,
    TrainingSnapshot,
    MonitorStatus,
    compute_gradient_histogram,
    detect_dead_neurons,
    compute_activation_distribution,
)


class TestGradientReport:
    """Test GradientReport dataclass."""

    def test_gradient_report_creation(self):
        """Test creating a gradient report."""
        report = GradientReport(
            layer_name="layer1.weight",
            mean=0.5,
            std=1.0,
            min_val=-2.0,
            max_val=3.0,
            norm=5.0,
            zero_ratio=0.1,
            nan_count=0,
            inf_count=0,
        )

        assert report.layer_name == "layer1.weight"
        assert report.mean == 0.5
        assert report.std == 1.0
        assert report.nan_count == 0

    def test_gradient_report_to_dict(self):
        """Test converting report to dictionary."""
        report = GradientReport(
            layer_name="test",
            mean=0.0,
            std=1.0,
            min_val=-1.0,
            max_val=1.0,
            norm=2.0,
            zero_ratio=0.0,
            nan_count=0,
            inf_count=0,
        )

        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["layer_name"] == "test"
        assert d["norm"] == 2.0


class TestActivationReport:
    """Test ActivationReport dataclass."""

    def test_activation_report_creation(self):
        """Test creating an activation report."""
        report = ActivationReport(
            layer_name="relu1",
            mean=0.5,
            std=0.3,
            min_val=0.0,
            max_val=2.0,
            zero_ratio=0.3,
            saturation_ratio=0.05,
            nan_count=0,
        )

        assert report.layer_name == "relu1"
        assert report.zero_ratio == 0.3
        assert report.saturation_ratio == 0.05


class TestWeightUpdateReport:
    """Test WeightUpdateReport dataclass."""

    def test_weight_update_report_creation(self):
        """Test creating a weight update report."""
        report = WeightUpdateReport(
            layer_name="fc1",
            weight_norm=10.0,
            update_norm=0.1,
            update_ratio=0.01,
            gradient_norm=0.05,
        )

        assert report.layer_name == "fc1"
        assert report.update_ratio == 0.01


class TestTrainingMonitor:
    """Test TrainingMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = TrainingMonitor()

        assert monitor._step == 0
        assert monitor._snapshots == []
        assert monitor.writer is None
        assert monitor.wandb is None

    def test_monitor_with_log_frequency(self):
        """Test monitor with custom log frequency."""
        monitor = TrainingMonitor(log_frequency=50)
        assert monitor.log_frequency == 50

    def test_record_gradients(self):
        """Test recording gradients."""
        monitor = TrainingMonitor()

        gradients = [
            ("layer1.weight", np.array([[0.5, -0.3], [0.2, 0.1]])),
            ("layer1.bias", np.array([0.1, -0.1])),
        ]

        reports = monitor.record_gradients(gradients, step=1)

        assert len(reports) == 2
        assert reports[0].layer_name == "layer1.weight"
        assert reports[0].norm > 0
        assert monitor._step == 1

    def test_record_activations(self):
        """Test recording activations."""
        monitor = TrainingMonitor()

        activations = np.array([[0.5, 0.0, 0.3], [0.1, 0.2, 0.0]])
        report = monitor.record_activations("relu1", activations, step=1)

        assert report.layer_name == "relu1"
        assert report.mean > 0
        assert report.zero_ratio > 0  # Some zeros present

    def test_record_weights(self):
        """Test recording weights."""
        monitor = TrainingMonitor()

        weights = [
            ("fc1.weight", np.random.randn(10, 5)),
            ("fc1.bias", np.zeros(5)),
        ]

        stats = monitor.record_weights(weights, step=1)

        assert "fc1.weight" in stats
        assert "fc1.bias" in stats
        assert stats["fc1.weight"]["norm"] > 0

    def test_compute_weight_update_ratio(self):
        """Test weight update ratio computation."""
        monitor = TrainingMonitor()

        prev_weight = np.array([[1.0, 2.0], [3.0, 4.0]])
        curr_weight = np.array([[1.01, 2.02], [3.01, 4.02]])
        gradient = np.array([[0.01, 0.02], [0.01, 0.02]])

        report = monitor.compute_weight_update_ratio(
            "fc1", curr_weight, prev_weight, gradient, learning_rate=0.01
        )

        assert report.layer_name == "fc1"
        assert report.update_ratio > 0
        assert report.update_ratio < 1  # Should be small

    def test_check_health_healthy(self):
        """Test health check with healthy gradients."""
        monitor = TrainingMonitor()

        reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1.0,
                min_val=-2.0,
                max_val=2.0,
                norm=1.0,
                zero_ratio=0.1,
                nan_count=0,
                inf_count=0,
            )
        ]

        status, warnings = monitor.check_health(reports)

        assert status == MonitorStatus.HEALTHY
        assert len(warnings) == 0

    def test_check_health_vanishing_gradient(self):
        """Test health check detects vanishing gradients."""
        monitor = TrainingMonitor()

        reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1e-10,
                min_val=-1e-10,
                max_val=1e-10,
                norm=1e-10,
                zero_ratio=0.1,
                nan_count=0,
                inf_count=0,
            )
        ]

        status, warnings = monitor.check_health(reports)

        assert status == MonitorStatus.WARNING
        assert any("Vanishing" in w for w in warnings)

    def test_check_health_exploding_gradient(self):
        """Test health check detects exploding gradients."""
        monitor = TrainingMonitor()

        reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1000.0,
                min_val=-1000.0,
                max_val=1000.0,
                norm=1000.0,
                zero_ratio=0.0,
                nan_count=0,
                inf_count=0,
            )
        ]

        status, warnings = monitor.check_health(reports)

        assert status == MonitorStatus.WARNING
        assert any("Exploding" in w for w in warnings)

    def test_check_health_nan_gradient(self):
        """Test health check detects NaN gradients."""
        monitor = TrainingMonitor()

        reports = [
            GradientReport(
                layer_name="layer1",
                mean=np.nan,
                std=1.0,
                min_val=-1.0,
                max_val=1.0,
                norm=1.0,
                zero_ratio=0.0,
                nan_count=5,
                inf_count=0,
            )
        ]

        status, warnings = monitor.check_health(reports)

        assert status == MonitorStatus.CRITICAL
        assert any("NaN" in w for w in warnings)

    def test_check_health_high_zero_ratio(self):
        """Test health check detects high zero ratio."""
        monitor = TrainingMonitor()

        reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=0.1,
                min_val=-0.5,
                max_val=0.5,
                norm=0.5,
                zero_ratio=0.95,  # 95% zeros
                nan_count=0,
                inf_count=0,
            )
        ]

        status, warnings = monitor.check_health(reports)

        assert status == MonitorStatus.WARNING
        assert any("zero ratio" in w for w in warnings)

    def test_check_health_weight_update_ratio_low(self):
        """Test health check detects low weight update ratio."""
        monitor = TrainingMonitor()

        gradient_reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1.0,
                min_val=-1.0,
                max_val=1.0,
                norm=1.0,
                zero_ratio=0.0,
                nan_count=0,
                inf_count=0,
            )
        ]

        weight_reports = [
            WeightUpdateReport(
                layer_name="layer1",
                weight_norm=10.0,
                update_norm=0.0001,
                update_ratio=0.00001,  # Very low
                gradient_norm=0.001,
            )
        ]

        status, warnings = monitor.check_health(gradient_reports, weight_update_reports=weight_reports)

        assert status == MonitorStatus.WARNING
        assert any("Low update ratio" in w for w in warnings)

    def test_check_health_weight_update_ratio_high(self):
        """Test health check detects high weight update ratio."""
        monitor = TrainingMonitor()

        gradient_reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1.0,
                min_val=-1.0,
                max_val=1.0,
                norm=1.0,
                zero_ratio=0.0,
                nan_count=0,
                inf_count=0,
            )
        ]

        weight_reports = [
            WeightUpdateReport(
                layer_name="layer1",
                weight_norm=1.0,
                update_norm=0.5,
                update_ratio=0.5,  # High (50%)
                gradient_norm=5.0,
            )
        ]

        status, warnings = monitor.check_health(gradient_reports, weight_update_reports=weight_reports)

        assert status == MonitorStatus.WARNING
        assert any("High update ratio" in w for w in warnings)

    def test_get_snapshot(self):
        """Test creating a training snapshot."""
        monitor = TrainingMonitor()

        gradient_reports = [
            GradientReport(
                layer_name="layer1",
                mean=0.0,
                std=1.0,
                min_val=-1.0,
                max_val=1.0,
                norm=1.0,
                zero_ratio=0.0,
                nan_count=0,
                inf_count=0,
            )
        ]

        snapshot = monitor.get_snapshot(
            step=10,
            loss=0.5,
            learning_rate=0.001,
            gradient_reports=gradient_reports,
        )

        assert snapshot.step == 10
        assert snapshot.loss == 0.5
        assert snapshot.learning_rate == 0.001
        assert snapshot.status == MonitorStatus.HEALTHY
        assert len(snapshot.gradient_reports) == 1

    def test_get_summary(self):
        """Test getting training summary."""
        monitor = TrainingMonitor()

        # Create some snapshots
        for i in range(5):
            reports = [
                GradientReport(
                    layer_name="layer1",
                    mean=0.0,
                    std=1.0,
                    min_val=-1.0,
                    max_val=1.0,
                    norm=1.0,
                    zero_ratio=0.0,
                    nan_count=0,
                    inf_count=0,
                )
            ]
            monitor.get_snapshot(step=i, loss=1.0 - i * 0.1, learning_rate=0.001, gradient_reports=reports)

        summary = monitor.get_summary()

        assert summary["total_steps"] == 5
        assert summary["latest_step"] == 4
        assert summary["latest_loss"] == pytest.approx(0.6, abs=0.01)
        assert summary["latest_status"] == "healthy"


class TestComputeGradientHistogram:
    """Test compute_gradient_histogram function."""

    def test_gradient_histogram_basic(self):
        """Test basic gradient histogram computation."""
        gradients = [
            np.array([0.1, 0.2, -0.1]),
            np.array([0.3, -0.2, 0.0]),
        ]

        hist = compute_gradient_histogram(gradients, bins=10)

        assert "counts" in hist
        assert "bin_edges" in hist
        assert "mean" in hist
        assert "std" in hist
        assert len(hist["counts"]) == 10

    def test_gradient_histogram_statistics(self):
        """Test gradient histogram statistics."""
        gradients = [
            np.array([1.0, 2.0, 3.0]),
        ]

        hist = compute_gradient_histogram(gradients)

        assert hist["mean"] == pytest.approx(2.0, abs=0.01)
        assert hist["min"] == pytest.approx(1.0, abs=0.01)
        assert hist["max"] == pytest.approx(3.0, abs=0.01)


class TestDetectDeadNeurons:
    """Test detect_dead_neurons function."""

    def test_no_dead_neurons(self):
        """Test with no dead neurons."""
        activation = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.4, 0.3],
            [0.2, 0.1, 0.5],
        ])

        has_dead, dead_ratio = detect_dead_neurons(activation, threshold=0.99)

        assert has_dead == False  # Using == instead of `is` for numpy bool
        assert dead_ratio == 0.0

    def test_with_dead_neurons(self):
        """Test with dead neurons."""
        # Create activation where first column is always zero
        activation = np.array([
            [0.0, 0.3, 0.2],
            [0.0, 0.4, 0.3],
            [0.0, 0.1, 0.5],
        ])

        has_dead, dead_ratio = detect_dead_neurons(activation, threshold=0.99)

        assert has_dead == True  # Using == instead of `is` for numpy bool
        assert dead_ratio > 0

    def test_conv_activation(self):
        """Test with convolution-style activation."""
        # (batch, channels, height, width)
        np.random.seed(42)
        activation = np.random.randn(4, 16, 8, 8)
        # Make half the channels dead
        activation[:, :8, :, :] = 0  # First 8 channels are "dead"

        has_dead, dead_ratio = detect_dead_neurons(activation, threshold=0.99)

        assert has_dead == True  # Should detect dead channels


class TestComputeActivationDistribution:
    """Test compute_activation_distribution function."""

    def test_relu_distribution(self):
        """Test ReLU activation distribution."""
        activation = np.array([[0.0, 0.5, 1.0], [0.0, 0.2, 0.8]])

        stats = compute_activation_distribution(activation, "relu")

        assert stats["mean"] > 0
        assert stats["zero_ratio"] == pytest.approx(1/3, abs=0.01)
        assert "positive_ratio" in stats

    def test_sigmoid_distribution(self):
        """Test sigmoid activation distribution."""
        # Simulate saturated sigmoid outputs
        activation = np.array([[0.01, 0.5, 0.99], [0.02, 0.6, 0.98]])

        stats = compute_activation_distribution(activation, "sigmoid")

        assert "saturation_low" in stats
        assert "saturation_high" in stats
        assert stats["saturation_low"] > 0  # Some low saturation
        assert stats["saturation_high"] > 0  # Some high saturation

    def test_tanh_distribution(self):
        """Test tanh activation distribution."""
        # Simulate saturated tanh outputs
        activation = np.array([[-0.99, 0.0, 0.99], [-0.98, 0.1, 0.98]])

        stats = compute_activation_distribution(activation, "tanh")

        assert "saturation_low" in stats
        assert "saturation_high" in stats


class TestTrainingSnapshot:
    """Test TrainingSnapshot dataclass."""

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = TrainingSnapshot(
            step=100,
            loss=0.5,
            learning_rate=0.001,
            gradient_reports=[],
            activation_reports=[],
            weight_update_reports=[],
            status=MonitorStatus.HEALTHY,
            warnings=[],
        )

        d = snapshot.to_dict()

        assert d["step"] == 100
        assert d["loss"] == 0.5
        assert d["learning_rate"] == 0.001
        assert d["status"] == "healthy"


class TestMonitorThresholds:
    """Test monitor threshold configurations."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        monitor = TrainingMonitor()

        assert monitor.GRADIENT_NORM_MIN == 1e-7
        assert monitor.GRADIENT_NORM_MAX == 100.0
        assert monitor.ZERO_RATIO_MAX == 0.9
        assert monitor.UPDATE_RATIO_MIN == 0.001
        assert monitor.UPDATE_RATIO_MAX == 0.1

    def test_update_ratio_in_healthy_range(self):
        """Test weight update ratio in healthy range (0.001-0.1)."""
        monitor = TrainingMonitor()

        # Update ratio of 0.01 should be healthy
        reports = [
            WeightUpdateReport(
                layer_name="layer1",
                weight_norm=10.0,
                update_norm=0.1,
                update_ratio=0.01,  # In healthy range
                gradient_norm=1.0,
            )
        ]

        status, warnings = monitor.check_health([], weight_update_reports=reports)

        # No weight update warnings should be triggered
        weight_warnings = [w for w in warnings if "update ratio" in w]
        assert len(weight_warnings) == 0


class TestMonitorEdgeCases:
    """Test edge cases in monitor."""

    def test_empty_gradient_list(self):
        """Test with empty gradient list."""
        monitor = TrainingMonitor()
        reports = monitor.record_gradients([])
        assert reports == []

    def test_none_gradient(self):
        """Test with None gradients (should be skipped)."""
        monitor = TrainingMonitor()

        gradients = [
            ("layer1.weight", np.array([1.0, 2.0])),
            ("layer1.bias", None),  # No gradient
        ]

        reports = monitor.record_gradients(gradients)

        assert len(reports) == 1
        assert reports[0].layer_name == "layer1.weight"

    def test_summary_with_no_snapshots(self):
        """Test summary with no snapshots collected."""
        monitor = TrainingMonitor()
        summary = monitor.get_summary()

        assert "error" in summary

    def test_weight_update_division_by_zero(self):
        """Test weight update with near-zero weight norm."""
        monitor = TrainingMonitor()

        # Very small weight norm
        report = monitor.compute_weight_update_ratio(
            "layer1",
            np.array([[1e-12, 1e-12]]),
            np.array([[0.0, 0.0]]),
            np.array([[0.1, 0.1]]),
            learning_rate=0.01,
        )

        # Should not raise, uses epsilon protection
        assert np.isfinite(report.update_ratio)
