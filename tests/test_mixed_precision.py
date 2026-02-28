"""
Tests for Mixed Precision Training module.

Tests include:
    - Precision detection utilities
    - GradScaler wrapper
    - MixedPrecisionTrainer
    - CPU fallback behavior
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from phase4_advanced.mixed_precision import (
    # Precision detection
    is_fp16_supported,
    is_bf16_supported,
    is_tf32_supported,
    enable_tf32,
    get_recommended_precision,
    get_device_info,
    # Scaler
    GradScalerConfig,
    MixedPrecisionScaler,
    # Trainer
    MixedPrecisionTrainer,
    # Utilities
    get_precision_info,
    MIXED_PRECISION_MODES,
    enable_optimizations_for_small_vram,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleModel()


@pytest.fixture
def simple_data():
    """Create simple data for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")

    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 5, (32,))
    return inputs, targets


# =============================================================================
# Test Precision Detection
# =============================================================================


class TestPrecisionDetection:
    """Test precision detection utilities."""

    def test_is_fp16_supported_returns_bool(self):
        """Test that is_fp16_supported returns a boolean."""
        result = is_fp16_supported()
        assert isinstance(result, bool)

    def test_is_bf16_supported_returns_bool(self):
        """Test that is_bf16_supported returns a boolean."""
        result = is_bf16_supported()
        assert isinstance(result, bool)

    def test_is_tf32_supported_returns_bool(self):
        """Test that is_tf32_supported returns a boolean."""
        result = is_tf32_supported()
        assert isinstance(result, bool)

    def test_get_recommended_precision_returns_string(self):
        """Test that get_recommended_precision returns a valid string."""
        result = get_recommended_precision()
        assert result in ('fp16', 'bf16', 'fp32')

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert 'cuda_available' in info
        assert 'fp16_supported' in info
        assert 'bf16_supported' in info

    def test_enable_tf32_does_not_raise(self):
        """Test that enable_tf32 doesn't raise errors."""
        # Should not raise even if TF32 not supported
        enable_tf32(True)
        enable_tf32(False)
        enable_tf32(True)


# =============================================================================
# Test GradScalerConfig
# =============================================================================


class TestGradScalerConfig:
    """Test GradScalerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GradScalerConfig()
        assert config.init_scale == 2.0 ** 16  # 65536
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.growth_interval == 2000
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GradScalerConfig(
            init_scale=1024.0,
            growth_factor=1.5,
            backoff_factor=0.3,
            growth_interval=1000,
            enabled=False,
        )
        assert config.init_scale == 1024.0
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.3
        assert config.growth_interval == 1000
        assert config.enabled is False


# =============================================================================
# Test MixedPrecisionScaler
# =============================================================================


class TestMixedPrecisionScaler:
    """Test MixedPrecisionScaler class."""

    def test_scaler_init(self):
        """Test scaler initialization."""
        scaler = MixedPrecisionScaler(
            init_scale=1024.0,
            enabled=True,
        )
        assert scaler._init_scale == 1024.0
        assert scaler.enabled == HAS_TORCH

    def test_scaler_scale_property(self):
        """Test scaler scale property."""
        scaler = MixedPrecisionScaler(init_scale=1024.0)
        scale = scaler.scale
        assert isinstance(scale, float)
        assert scale > 0

    def test_scaler_get_stats(self):
        """Test scaler statistics."""
        scaler = MixedPrecisionScaler()
        stats = scaler.get_stats()
        assert isinstance(stats, dict)
        assert 'current_scale' in stats
        assert 'step_count' in stats
        assert 'found_inf_count' in stats
        assert 'enabled' in stats

    def test_scaler_health_check(self):
        """Test scaler health check."""
        scaler = MixedPrecisionScaler()
        # Should pass with default scale
        assert scaler.is_health_check_passed(min_scale=1.0)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_scaler_with_torch_tensor(self):
        """Test scaler with PyTorch tensor."""
        scaler = MixedPrecisionScaler(enabled=False)

        # Create a simple loss tensor
        loss = torch.tensor(0.5, requires_grad=True)

        # Scale loss (should return same tensor when disabled)
        scaled_loss = scaler.scale_loss(loss)
        assert scaled_loss.item() == 0.5


# =============================================================================
# Test MixedPrecisionTrainer
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestMixedPrecisionTrainer:
    """Test MixedPrecisionTrainer class."""

    def test_trainer_init(self, simple_model):
        """Test trainer initialization."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        assert trainer.precision == 'fp32'
        assert trainer.use_amp is False
        assert trainer.device is not None

    def test_trainer_auto_precision(self, simple_model):
        """Test trainer auto precision selection."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='auto',
        )

        # Should select a valid precision
        assert trainer.precision in ('fp16', 'bf16', 'fp32')

    def test_trainer_train_step(self, simple_model, simple_data):
        """Test trainer train step."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        inputs, targets = simple_data
        loss = trainer.train_step(inputs, targets)

        assert isinstance(loss, float)
        assert loss > 0
        assert len(trainer._loss_history) == 1

    def test_trainer_eval_step(self, simple_model, simple_data):
        """Test trainer eval step."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        inputs, targets = simple_data
        loss, outputs = trainer.eval_step(inputs, targets)

        assert isinstance(loss, float)
        assert outputs.shape == (32, 5)

    def test_trainer_get_stats(self, simple_model, simple_data):
        """Test trainer statistics."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        inputs, targets = simple_data
        trainer.train_step(inputs, targets)

        stats = trainer.get_training_stats()
        assert stats['step_count'] == 1
        assert len(stats['loss_history']) == 1
        assert 'precision' in stats
        assert 'device' in stats

    def test_trainer_with_grad_clip(self, simple_model, simple_data):
        """Test trainer with gradient clipping."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
            grad_clip_norm=1.0,
        )

        inputs, targets = simple_data
        loss = trainer.train_step(inputs, targets)

        assert isinstance(loss, float)


# =============================================================================
# Test Precision Info
# =============================================================================


class TestPrecisionInfo:
    """Test precision info utilities."""

    def test_get_precision_info_fp16(self):
        """Test getting FP16 info."""
        info = get_precision_info('fp16')
        assert info['dtype'] == 'float16'
        assert info['bytes_per_element'] == 2

    def test_get_precision_info_fp32(self):
        """Test getting FP32 info."""
        info = get_precision_info('fp32')
        assert info['dtype'] == 'float32'
        assert info['bytes_per_element'] == 4

    def test_get_precision_info_invalid(self):
        """Test getting info for invalid precision."""
        with pytest.raises(ValueError):
            get_precision_info('invalid')

    def test_mixed_precision_modes_dict(self):
        """Test MIXED_PRECISION_MODES dictionary."""
        assert 'fp16' in MIXED_PRECISION_MODES
        assert 'bf16' in MIXED_PRECISION_MODES
        assert 'fp32' in MIXED_PRECISION_MODES
        assert 'tf32' in MIXED_PRECISION_MODES


# =============================================================================
# Test Small VRAM Optimizations
# =============================================================================


class TestSmallVRAMOptimizations:
    """Test optimizations for small VRAM."""

    def test_enable_optimizations_returns_dict(self):
        """Test that enable_optimizations_for_small_vram returns dict."""
        opts = enable_optimizations_for_small_vram()
        assert isinstance(opts, dict)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_optimizations_on_cpu(self):
        """Test optimizations on CPU (should not crash)."""
        # Should not crash even on CPU
        opts = enable_optimizations_for_small_vram()
        assert isinstance(opts, dict)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestIntegration:
    """Integration tests for mixed precision training."""

    def test_training_loop(self, simple_model):
        """Test complete training loop."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        # Create data
        inputs = torch.randn(100, 10)
        targets = torch.randint(0, 5, (100,))
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train for multiple steps
        losses = []
        for batch_inputs, batch_targets in dataloader:
            loss = trainer.train_step(batch_inputs, batch_targets)
            losses.append(loss)

        # Check that training progressed
        assert len(losses) > 0
        assert all(isinstance(l, float) for l in losses)

    def test_convergence(self, simple_model):
        """Test that training converges."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = MixedPrecisionTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            precision='fp32',
        )

        # Create simple linearly separable data
        inputs = torch.randn(100, 10)
        targets = (inputs[:, 0] > 0).long()
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train for several epochs
        initial_loss = None
        final_loss = None

        for epoch in range(3):
            for batch_inputs, batch_targets in dataloader:
                loss = trainer.train_step(batch_inputs, batch_targets)
                if initial_loss is None:
                    initial_loss = loss
                final_loss = loss

        # Loss should decrease
        assert final_loss < initial_loss


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
