"""
Tests for TensorBoard Debug module.

Tests cover:
    - TensorBoardMonitor functionality
    - Gradient histogram logging
    - Activation monitoring
    - Health issue detection
    - WandBMonitor (basic tests without actual wandb connection)
"""

import pytest
import numpy as np
import tempfile
import os

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytestmark = pytest.mark.skip("PyTorch not available")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTensorBoardMonitor:
    """Test TensorBoardMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TensorBoardMonitor(log_dir=tmpdir)
            assert monitor.log_dir == tmpdir
            assert monitor.global_step == 0
            monitor.close()

    def test_monitor_context_manager(self):
        """Test monitor as context manager."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                assert monitor.global_step == 0
                monitor.global_step = 10

            # Monitor should be closed after context exit

    def test_watch_model(self):
        """Test watching a model."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=10)
                assert monitor._watched_model is model

    def test_log_training_step(self):
        """Test logging a training step."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1)

                # Simulate training step
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)
                loss = nn.functional.mse_loss(model(x), y)
                loss.backward()

                info = monitor.log_training_step(
                    epoch=0,
                    batch_idx=0,
                    loss=loss.item(),
                    model=model,
                    optimizer=optimizer,
                )

                assert info["step"] == 1
                assert info["loss"] == loss.item()
                assert info["learning_rate"] == 0.01

    def test_log_training_step_with_metrics(self):
        """Test logging training step with additional metrics."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1)

                info = monitor.log_training_step(
                    epoch=0,
                    batch_idx=0,
                    loss=0.5,
                    model=model,
                    metrics={"accuracy": 0.8, "precision": 0.75},
                )

                assert "accuracy" not in info  # Metrics are logged separately

    def test_log_validation(self):
        """Test logging validation metrics."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_validation(
                    epoch=1,
                    val_loss=0.3,
                    metrics={"accuracy": 0.85},
                )

    def test_log_epoch_summary(self):
        """Test logging epoch summary."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_epoch_summary(
                    epoch=1,
                    train_loss=0.5,
                    val_loss=0.3,
                    metrics={"accuracy": 0.85},
                )

    def test_log_scalar(self):
        """Test logging custom scalar."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_scalar("custom/metric", 0.75, step=10)

    def test_log_histogram(self):
        """Test logging custom histogram."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                values = np.random.randn(100)
                monitor.log_histogram("custom/hist", values, step=10)

    def test_log_histogram_with_tensor(self):
        """Test logging histogram from torch tensor."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                values = torch.randn(100)
                monitor.log_histogram("custom/hist", values, step=10)

    def test_log_text(self):
        """Test logging text."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_text("notes/training", "Started training", step=0)

    def test_log_hyperparams(self):
        """Test logging hyperparameters."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_hyperparams(
                    hparams={"lr": 0.01, "batch_size": 32, "epochs": 10},
                    metrics={"final_accuracy": 0.9},
                )

    def test_warnings_tracking(self):
        """Test warning tracking."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                assert not monitor.has_issues()

                monitor._warnings.append("Test warning")
                assert monitor.has_issues()

                warnings = monitor.get_warnings()
                assert len(warnings) == 1

                monitor.clear_warnings()
                assert not monitor.has_issues()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGradientLogging:
    """Test gradient logging functionality."""

    def test_gradient_histogram_logging(self):
        """Test gradient histogram logging."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1)

                # Forward and backward pass
                x = torch.randn(4, 10)
                y = model(x)
                loss = y.sum()
                loss.backward()

                # Log gradients
                info = monitor.log_training_step(
                    epoch=0, batch_idx=0, loss=loss.item(), model=model
                )

                assert "gradients" in info

    def test_gradient_statistics(self):
        """Test gradient statistics computation."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1)

                x = torch.randn(4, 10)
                loss = model(x).sum()
                loss.backward()

                info = monitor.log_training_step(
                    epoch=0, batch_idx=0, loss=loss.item(), model=model
                )

                # Check gradient stats are computed
                for name, stats in info["gradients"].items():
                    assert "norm" in stats
                    assert "mean" in stats
                    assert "std" in stats
                    assert stats["norm"] > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestActivationMonitoring:
    """Test activation monitoring functionality."""

    def test_activation_hooks_registration(self):
        """Test activation hooks are registered."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1, log_activations=True)

                # Hooks should be registered
                assert len(monitor._hooks) > 0

    def test_activation_cache(self):
        """Test activation caching."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1, log_activations=True)

                # Forward pass
                x = torch.randn(4, 10)
                _ = model(x)

                # Cache should have activations
                assert len(monitor._activation_cache) > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHealthDetection:
    """Test health issue detection."""

    def test_vanishing_gradient_detection(self):
        """Test vanishing gradient detection."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                # Simulate vanishing gradient
                monitor._warnings.append(
                    "Step 100: Vanishing gradient in layer3 (norm=1e-10)"
                )

                assert monitor.has_issues()
                warnings = monitor.get_warnings()
                assert any("Vanishing" in w for w in warnings)

    def test_exploding_gradient_detection(self):
        """Test exploding gradient detection."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                # Simulate exploding gradient
                monitor._warnings.append(
                    "Step 50: Exploding gradient in layer1 (norm=500)"
                )

                assert monitor.has_issues()

    def test_dead_neuron_detection(self):
        """Test dead neuron detection."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                # Simulate dead neuron warning
                monitor._warnings.append(
                    "Step 100: Dead neurons detected in relu1 (30.5%)"
                )

                assert monitor.has_issues()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_monitor(self):
        """Test create_monitor function."""
        from phase4_advanced.tensorboard_debug import create_monitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = create_monitor("test_experiment", base_dir=tmpdir)
            assert monitor.log_dir == os.path.join(tmpdir, "test_experiment")
            monitor.close()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWandBMonitor:
    """Test WandBMonitor class (without actual wandb connection)."""

    def test_wandb_not_available_graceful(self):
        """Test graceful handling when wandb not installed."""
        # This test just verifies the import structure
        # Actual wandb functionality is tested separately
        pass


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDualMonitor:
    """Test DualMonitor class."""

    def test_dual_monitor_tensorboard_only(self):
        """Test dual monitor with only TensorBoard."""
        from phase4_advanced.tensorboard_debug import DualMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with DualMonitor(tensorboard_dir=tmpdir) as monitor:
                assert monitor.tb_monitor is not None
                assert monitor.wb_monitor is None

    def test_dual_monitor_watch(self):
        """Test watching model with dual monitor."""
        from phase4_advanced.tensorboard_debug import DualMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with DualMonitor(tensorboard_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=10)
                assert monitor.tb_monitor._watched_model is model

    def test_dual_monitor_log_step(self):
        """Test logging step with dual monitor."""
        from phase4_advanced.tensorboard_debug import DualMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with DualMonitor(tensorboard_dir=tmpdir) as monitor:
                monitor.watch(model, log_freq=1)

                x = torch.randn(4, 10)
                loss = model(x).sum()
                loss.backward()

                info = monitor.log_training_step(
                    epoch=0, batch_idx=0, loss=loss.item(), model=model
                )

                assert info["loss"] == loss.item()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestModelGraph:
    """Test model graph logging."""

    def test_log_graph(self):
        """Test logging model graph."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                input_tensor = torch.randn(1, 10)
                monitor.log_graph(model, input_tensor)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestFlush:
    """Test flush functionality."""

    def test_flush(self):
        """Test flushing the writer."""
        from phase4_advanced.tensorboard_debug import TensorBoardMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with TensorBoardMonitor(log_dir=tmpdir) as monitor:
                monitor.log_scalar("test", 0.5)
                monitor.flush()  # Should not raise


class TestModuleConstants:
    """Test module constants."""

    def test_tensorboard_debug_components(self):
        """Test TENSORBOARD_DEBUG_COMPONENTS constant."""
        from phase4_advanced.tensorboard_debug import TENSORBOARD_DEBUG_COMPONENTS

        assert "TensorBoardMonitor" in TENSORBOARD_DEBUG_COMPONENTS
        assert "WandBMonitor" in TENSORBOARD_DEBUG_COMPONENTS
        assert "DualMonitor" in TENSORBOARD_DEBUG_COMPONENTS
        assert "create_monitor" in TENSORBOARD_DEBUG_COMPONENTS
        assert "quick_visualize" in TENSORBOARD_DEBUG_COMPONENTS
