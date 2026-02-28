"""
Unit tests for Gradient Accumulation implementation.

Tests cover:
    - GradientAccumulator: Core accumulation logic
    - GradientAccumulationTrainer: Training loop with accumulation
    - Memory utilities
    - Gradient equivalence verification
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import gradient accumulation module
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch not available", allow_module_level=True)

from phase5_deployment.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulator,
    GradientAccumulationTrainer,
    get_memory_usage,
    reset_memory_stats,
    benchmark_memory_usage,
    verify_gradient_equivalence,
    calculate_memory_savings,
    create_gradient_accumulation_trainer,
    recommend_accumulation_settings,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleMLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleMLP()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    torch.manual_seed(42)
    # Create dummy MNIST-like data
    x = torch.randn(256, 1, 28, 28)
    y = torch.randint(0, 10, (256,))
    return x, y


@pytest.fixture
def dataloader(sample_data):
    """Create DataLoader for testing."""
    x, y = sample_data
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)


# =============================================================================
# GradientAccumulationConfig Tests
# =============================================================================


class TestGradientAccumulationConfig:
    """Tests for GradientAccumulationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GradientAccumulationConfig()
        assert config.accumulation_steps == 4
        assert config.batch_size == 32
        assert config.effective_batch_size == 128
        assert config.use_amp is True
        assert config.max_grad_norm == 1.0

    def test_effective_batch_size_calculation(self):
        """Test effective batch size is calculated correctly."""
        config = GradientAccumulationConfig(
            batch_size=16,
            accumulation_steps=8,
        )
        assert config.effective_batch_size == 128

    def test_accumulation_steps_from_effective_batch(self):
        """Test accumulation steps derived from effective batch size."""
        config = GradientAccumulationConfig(
            batch_size=32,
            effective_batch_size=128,
        )
        assert config.accumulation_steps == 4

    def test_invalid_effective_batch_size(self):
        """Test error when effective batch not divisible by batch size."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(
                batch_size=32,
                effective_batch_size=100,  # Not divisible
            )

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = GradientAccumulationConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_accumulation_steps(self):
        """Test validation fails for invalid accumulation steps."""
        config = GradientAccumulationConfig(accumulation_steps=0)
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_negative_max_grad_norm(self):
        """Test validation fails for negative max grad norm."""
        config = GradientAccumulationConfig(max_grad_norm=-1.0)
        errors = config.validate()
        assert len(errors) > 0


# =============================================================================
# GradientAccumulator Tests
# =============================================================================


class TestGradientAccumulator:
    """Tests for GradientAccumulator."""

    def test_initialization(self):
        """Test accumulator initializes correctly."""
        acc = GradientAccumulator(accumulation_steps=4)
        assert acc.accumulation_steps == 4
        assert acc.step_count == 0

    def test_invalid_accumulation_steps(self):
        """Test error for invalid accumulation steps."""
        with pytest.raises(ValueError):
            GradientAccumulator(accumulation_steps=0)

    def test_should_update_timing(self):
        """Test should_update returns True at correct intervals."""
        acc = GradientAccumulator(accumulation_steps=4)

        # Steps 0, 1, 2 should not update
        for i in range(3):
            assert acc.step_count == i
            assert not acc.should_update()
            acc.advance()

        # Step 3 should update
        assert acc.step_count == 3
        assert acc.should_update()

    def test_accumulate_context_manager(self):
        """Test context manager increments step count."""
        acc = GradientAccumulator(accumulation_steps=4)

        assert acc.step_count == 0

        with acc.accumulate():
            pass

        assert acc.step_count == 1

    def test_is_first_step(self):
        """Test is_first_step property."""
        acc = GradientAccumulator(accumulation_steps=4)

        assert acc.is_first_step is True
        acc.advance()
        assert acc.is_first_step is False

    def test_is_last_step(self):
        """Test is_last_step property."""
        acc = GradientAccumulator(accumulation_steps=4)

        for i in range(3):
            assert acc.is_last_step is False
            acc.advance()

        assert acc.is_last_step is True

    def test_reset(self):
        """Test reset resets step count."""
        acc = GradientAccumulator(accumulation_steps=4)

        for _ in range(3):
            acc.advance()

        assert acc.step_count == 3
        acc.reset()
        assert acc.step_count == 0

    def test_set_accumulation_steps(self):
        """Test dynamic adjustment of accumulation steps."""
        acc = GradientAccumulator(accumulation_steps=4)
        acc.advance()  # step_count = 1

        acc.set_accumulation_steps(8)
        assert acc.accumulation_steps == 8
        assert acc.step_count == 1  # Preserved modulo new steps

    def test_set_invalid_accumulation_steps(self):
        """Test error for invalid dynamic adjustment."""
        acc = GradientAccumulator(accumulation_steps=4)

        with pytest.raises(ValueError):
            acc.set_accumulation_steps(0)


# =============================================================================
# GradientAccumulationTrainer Tests
# =============================================================================


class TestGradientAccumulationTrainer:
    """Tests for GradientAccumulationTrainer."""

    def test_initialization(self, simple_model):
        """Test trainer initializes correctly."""
        config = GradientAccumulationConfig(accumulation_steps=4, batch_size=32)
        trainer = GradientAccumulationTrainer(simple_model, config)

        assert trainer.config.accumulation_steps == 4
        assert trainer.accumulator.accumulation_steps == 4
        assert trainer.global_step == 0

    def test_train_epoch(self, simple_model, dataloader):
        """Test single epoch training."""
        config = GradientAccumulationConfig(
            accumulation_steps=4,
            batch_size=32,
            use_amp=False,  # Disable AMP for CPU compatibility
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        assert 'loss' in metrics
        assert 'updates' in metrics
        assert 'samples' in metrics
        assert metrics['loss'] > 0
        # With 256 samples, batch_size=32, we have 8 batches
        # With accumulation_steps=4, we get 2 updates
        assert metrics['updates'] == 2

    def test_full_training(self, simple_model, dataloader):
        """Test full training loop."""
        config = GradientAccumulationConfig(
            accumulation_steps=4,
            batch_size=32,
            use_amp=False,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        history = trainer.train(
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=2,
        )

        assert len(history['train_loss']) == 2
        assert history['train_loss'][1] < history['train_loss'][0]  # Loss decreases

    def test_evaluate(self, simple_model, dataloader):
        """Test evaluation."""
        config = GradientAccumulationConfig(use_amp=False)
        trainer = GradientAccumulationTrainer(simple_model, config)

        criterion = nn.CrossEntropyLoss()
        metrics = trainer.evaluate(dataloader, criterion)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_set_accumulation_steps(self, simple_model):
        """Test dynamic accumulation steps adjustment."""
        config = GradientAccumulationConfig(accumulation_steps=4)
        trainer = GradientAccumulationTrainer(simple_model, config)

        trainer.set_accumulation_steps(8)

        assert trainer.accumulator.accumulation_steps == 8
        assert trainer.config.accumulation_steps == 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_amp_training(self, simple_model, dataloader):
        """Test AMP training on GPU."""
        config = GradientAccumulationConfig(
            accumulation_steps=4,
            use_amp=True,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(dataloader, optimizer, criterion)
        assert metrics['loss'] > 0


# =============================================================================
# Memory Utilities Tests
# =============================================================================


class TestMemoryUtilities:
    """Tests for memory utility functions."""

    def test_get_memory_usage_no_cuda(self):
        """Test memory usage when CUDA not available."""
        if not torch.cuda.is_available():
            mem = get_memory_usage()
            assert mem['allocated'] == 0
            assert mem['reserved'] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_memory_usage_cuda(self):
        """Test memory usage with CUDA."""
        mem = get_memory_usage('cuda:0')
        assert 'allocated' in mem
        assert 'reserved' in mem
        assert 'max_allocated' in mem
        assert mem['allocated'] >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_memory_stats(self):
        """Test memory stats reset."""
        reset_memory_stats('cuda:0')
        mem = get_memory_usage('cuda:0')
        assert mem['max_allocated'] >= 0

    def test_calculate_memory_savings(self):
        """Test memory savings calculation."""
        result = calculate_memory_savings(
            original_batch_size=128,
            accumulation_steps=4,
            model_activation_memory_per_sample=10.0,
        )

        assert result['original_memory_mb'] == 1280.0
        assert result['accumulated_memory_mb'] == 320.0
        assert result['savings_mb'] == 960.0
        assert result['savings_percent'] == 75.0


# =============================================================================
# Gradient Equivalence Tests
# =============================================================================


class TestGradientEquivalence:
    """Tests for gradient equivalence verification."""

    def test_equivalence_simple_model(self):
        """Test gradient equivalence with simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        result = verify_gradient_equivalence(
            model_class=SimpleModel,
            input_shape=(1, 28, 28),
            effective_batch_size=128,
            accumulation_steps=4,
            device='cpu',
        )

        assert result['equivalent'] is True
        assert result['max_difference'] < 1e-5

    def test_equivalence_with_different_steps(self):
        """Test equivalence with different accumulation steps."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        for steps in [2, 4, 8]:
            result = verify_gradient_equivalence(
                model_class=SimpleModel,
                input_shape=(1, 28, 28),
                effective_batch_size=64,
                accumulation_steps=steps,
                device='cpu',
            )

            assert result['equivalent'] is True, f"Failed with {steps} steps"


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_gradient_accumulation_trainer(self, simple_model):
        """Test trainer creation convenience function."""
        trainer = create_gradient_accumulation_trainer(
            model=simple_model,
            effective_batch_size=128,
            micro_batch_size=32,
            use_amp=False,
        )

        assert isinstance(trainer, GradientAccumulationTrainer)
        assert trainer.config.effective_batch_size == 128
        assert trainer.config.batch_size == 32
        assert trainer.config.accumulation_steps == 4

    def test_recommend_accumulation_settings(self):
        """Test setting recommendations."""
        result = recommend_accumulation_settings(
            vram_gb=4.0,
            model_params_m=10.0,
            input_size_kb=150.0,
        )

        assert 'recommended_max_batch_size' in result
        assert 'recommendations' in result
        assert result['recommended_max_batch_size'] > 0

    def test_recommend_accumulation_settings_low_vram(self):
        """Test recommendations for low VRAM."""
        result = recommend_accumulation_settings(
            vram_gb=2.0,
            model_params_m=50.0,
            input_size_kb=200.0,
        )

        # Should recommend small batch sizes
        assert result['recommended_max_batch_size'] >= 1


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestBenchmarks:
    """Tests for benchmarking functions."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_memory_usage(self, simple_model):
        """Test memory usage benchmarking."""
        sample_input = torch.randn(1, 1, 28, 28)

        result = benchmark_memory_usage(
            model=simple_model,
            sample_input=sample_input,
            batch_sizes=[4, 8, 16],
            accumulation_steps=[1, 2, 4],
            criterion=nn.CrossEntropyLoss(),
            device='cuda:0',
        )

        assert 'results' in result
        assert len(result['results']) == 9  # 3 batch_sizes x 3 accumulation_steps

        # Check that effective batch size is calculated correctly
        for r in result['results']:
            assert r['effective_batch_size'] == r['batch_size'] * r['accumulation_steps']


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for gradient accumulation."""

    def test_training_convergence(self, simple_model):
        """Test that training with accumulation converges."""
        # Create training data
        torch.manual_seed(42)
        x = torch.randn(500, 1, 28, 28)
        y = torch.randint(0, 10, (500,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=25, shuffle=True)

        config = GradientAccumulationConfig(
            accumulation_steps=4,
            batch_size=25,
            effective_batch_size=100,
            use_amp=False,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        history = trainer.train(
            train_loader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=5,
        )

        # Loss should decrease over training
        assert history['train_loss'][-1] < history['train_loss'][0]

    def test_dynamic_accumulation_during_training(self, simple_model):
        """Test dynamic accumulation adjustment during training."""
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=25, shuffle=False)

        config = GradientAccumulationConfig(
            accumulation_steps=4,
            use_amp=False,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train one epoch with accumulation_steps=4
        trainer.train_epoch(dataloader, optimizer, criterion)

        # Change accumulation steps
        trainer.set_accumulation_steps(2)

        # Train another epoch - should still work
        metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        assert metrics['updates'] > 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_accumulation_steps_one(self, simple_model, dataloader):
        """Test with accumulation_steps=1 (no accumulation)."""
        config = GradientAccumulationConfig(
            accumulation_steps=1,
            use_amp=False,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        # With no accumulation, updates = number of batches
        assert metrics['updates'] == len(dataloader)

    def test_large_accumulation_steps(self, simple_model):
        """Test with large accumulation steps exceeding batches."""
        # Small dataset
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        config = GradientAccumulationConfig(
            accumulation_steps=10,  # More than number of batches
            use_amp=False,
        )
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        # Should handle gracefully - 0 updates if accumulation not complete
        # This is expected behavior

    def test_empty_dataloader(self, simple_model):
        """Test with empty dataloader."""
        # Empty dataset
        x = torch.randn(0, 1, 28, 28)
        y = torch.randint(0, 10, (0,))
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=4)

        config = GradientAccumulationConfig(use_amp=False)
        trainer = GradientAccumulationTrainer(simple_model, config)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(dataloader, optimizer, criterion)

        assert metrics['loss'] == 0
        assert metrics['samples'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
