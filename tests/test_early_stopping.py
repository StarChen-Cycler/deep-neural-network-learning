"""
Tests for Early Stopping Callback.

Run with: pytest tests/test_early_stopping.py -v
"""

import pytest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_advanced.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
    EarlyStoppingTrainer,
    create_early_stopping,
    validate_early_stopping,
    EARLY_STOPPING_COMPONENTS,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    pytest.skip("PyTorch not available", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return TensorDataset(x, y)


@pytest.fixture
def simple_dataloader(simple_dataset):
    """Create a simple dataloader for testing."""
    return DataLoader(simple_dataset, batch_size=10)


# =============================================================================
# Test EarlyStoppingConfig
# =============================================================================


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EarlyStoppingConfig()
        assert config.patience == 5
        assert config.min_delta == 0.0
        assert config.mode == 'min'
        assert config.restore_best_weights == True
        assert config.verbose == True
        assert config.checkpoint_path is None
        assert config.save_checkpoint == False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EarlyStoppingConfig(
            patience=10,
            min_delta=0.001,
            mode='max',
            restore_best_weights=False,
            verbose=False,
            checkpoint_path='checkpoint.pt',
            save_checkpoint=True,
        )
        assert config.patience == 10
        assert config.min_delta == 0.001
        assert config.mode == 'max'
        assert config.restore_best_weights == False
        assert config.verbose == False
        assert config.checkpoint_path == 'checkpoint.pt'
        assert config.save_checkpoint == True


# =============================================================================
# Test EarlyStopping Callback
# =============================================================================


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_init_default(self):
        """Test default initialization."""
        es = EarlyStopping()
        assert es.patience == 5
        assert es.min_delta == 0.0
        assert es.mode == 'min'
        assert es.counter == 0
        assert es.early_stop == False

    def test_init_custom(self):
        """Test custom initialization."""
        es = EarlyStopping(patience=10, min_delta=0.01, mode='max')
        assert es.patience == 10
        assert es.min_delta == 0.01
        assert es.mode == 'max'

    def test_init_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            EarlyStopping(mode='invalid')

    def test_init_from_config(self):
        """Test initialization from config object."""
        config = EarlyStoppingConfig(patience=7, min_delta=0.005)
        es = EarlyStopping(config=config)
        assert es.patience == 7
        assert es.min_delta == 0.005

    def test_improvement_min_mode(self):
        """Test improvement detection in min mode."""
        es = EarlyStopping(mode='min', patience=5, verbose=False)

        # First call should improve (infinity -> any value)
        result = es.step(1.0)
        assert result == False
        assert es.counter == 0
        assert es.best_epoch == 1

        # Better value should improve
        result = es.step(0.5)
        assert result == False
        assert es.counter == 0
        assert es.best_epoch == 2

        # Same or worse value should not improve
        result = es.step(0.5)
        assert result == False
        assert es.counter == 1

        # Worse value should not improve
        result = es.step(0.6)
        assert result == False
        assert es.counter == 2

    def test_improvement_max_mode(self):
        """Test improvement detection in max mode."""
        es = EarlyStopping(mode='max', patience=5, verbose=False)

        # First call should improve (-infinity -> any value)
        result = es.step(0.5)
        assert result == False
        assert es.counter == 0

        # Better value should improve
        result = es.step(0.8)
        assert result == False
        assert es.counter == 0

        # Same or worse value should not improve
        result = es.step(0.7)
        assert result == False
        assert es.counter == 1

    def test_patience_triggers_stop(self, simple_model):
        """Test that patience triggers early stopping."""
        es = EarlyStopping(patience=3, verbose=False)

        # First improvement
        es.step(1.0, simple_model)

        # No improvement for patience steps
        for i in range(3):
            result = es.step(1.0, simple_model)  # Same value, no improvement

        # Should trigger after patience exhausted
        assert es.early_stop == True
        assert es.counter == 3

    def test_patience_5_triggers_stop(self, simple_model):
        """Test that patience=5 triggers early stopping (criterion)."""
        es = EarlyStopping(patience=5, verbose=False)

        # First improvement
        es.step(1.0, simple_model)

        # No improvement for 5 steps
        for i in range(5):
            result = es.step(1.0 + i * 0.01, simple_model)

        # Should trigger
        assert es.early_stop == True
        assert es.counter >= 5

    def test_best_weights_saved(self, simple_model):
        """Test that best weights are saved."""
        es = EarlyStopping(patience=5, restore_best_weights=True, verbose=False)

        # Initial weights
        initial_weight = simple_model.weight.clone()

        # First improvement
        es.step(1.0, simple_model)
        assert es.best_weights is not None

        # Modify model weights
        with torch.no_grad():
            simple_model.weight.fill_(0.0)

        # Best weights should still be original
        saved_weight = es.best_weights['weight']
        assert torch.allclose(saved_weight, initial_weight)

    def test_best_weights_at_lowest_loss(self, simple_model):
        """Test weights saved at lowest validation loss (criterion)."""
        es = EarlyStopping(patience=5, restore_best_weights=True, verbose=False)

        # First epoch - loss 1.0
        es.step(1.0, simple_model)
        weights_epoch1 = es.best_weights['weight'].clone()

        # Second epoch - loss 0.5 (better)
        es.step(0.5, simple_model)
        weights_epoch2 = es.best_weights['weight'].clone()

        # Third epoch - loss 0.8 (worse, no improvement)
        es.step(0.8, simple_model)

        # Best weights should be from epoch 2 (loss 0.5)
        assert es.best_epoch == 2
        assert torch.allclose(es.best_weights['weight'], weights_epoch2)

    def test_restore_weights(self, simple_model):
        """Test restoring best weights."""
        es = EarlyStopping(patience=5, restore_best_weights=True, verbose=False)

        # Save initial weights
        initial_weight = simple_model.weight.clone()

        # First improvement
        es.step(1.0, simple_model)

        # Modify model
        with torch.no_grad():
            simple_model.weight.fill_(0.5)

        # Restore
        es.restore_weights(simple_model)

        # Should be back to initial
        assert torch.allclose(simple_model.weight, initial_weight)

    def test_call_method(self, simple_model):
        """Test __call__ method works same as step."""
        es = EarlyStopping(patience=5, verbose=False)

        # Using call method
        result1 = es(1.0, simple_model)
        result2 = es.step(0.5, simple_model)

        assert result1 == False
        assert result2 == False
        assert es.best_epoch == 2

    def test_min_delta_threshold(self):
        """Test min_delta threshold for improvement."""
        es = EarlyStopping(patience=5, min_delta=0.1, verbose=False)

        # First improvement
        es.step(1.0)

        # Small improvement (below min_delta) should not count
        es.step(0.95)  # Only 0.05 improvement
        assert es.counter == 1

        # Large enough improvement
        es.step(0.8)  # 0.15 improvement
        assert es.counter == 0

    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=5, verbose=False)

        # Trigger some state
        es.step(1.0)
        es.step(1.5)
        es.step(2.0)

        assert es.counter > 0

        # Reset
        es.reset()

        assert es.counter == 0
        assert es.early_stop == False
        assert es.best_epoch == 0

    def test_get_state(self):
        """Test get_state serialization."""
        es = EarlyStopping(patience=10, min_delta=0.01)
        es.step(1.0)
        es.step(0.5)

        state = es.get_state()

        assert state['patience'] == 10
        assert state['min_delta'] == 0.01
        assert state['counter'] == 0
        assert state['best_epoch'] == 2

    def test_load_state(self):
        """Test load_state deserialization."""
        es1 = EarlyStopping(patience=10, min_delta=0.01)
        es1.step(1.0)
        es1.step(0.5)
        state = es1.get_state()

        es2 = EarlyStopping()
        es2.load_state(state)

        assert es2.patience == 10
        assert es2.min_delta == 0.01
        assert es2.best_epoch == 2

    def test_checkpoint_save(self, simple_model):
        """Test checkpoint saving on improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'best_model.pt')
            es = EarlyStopping(
                patience=5,
                checkpoint_path=checkpoint_path,
                save_checkpoint=True,
                verbose=False,
            )

            es.step(1.0, simple_model)

            # Checkpoint should be saved
            assert os.path.exists(checkpoint_path)

            # Load and verify
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert 'model_state_dict' in checkpoint
            assert 'best_min' in checkpoint

    def test_repr(self):
        """Test string representation."""
        es = EarlyStopping(patience=5, mode='min')
        repr_str = repr(es)
        assert 'EarlyStopping' in repr_str
        assert 'patience=5' in repr_str


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateEarlyStopping:
    """Tests for create_early_stopping factory."""

    def test_create_default(self):
        """Test default creation."""
        es = create_early_stopping()
        assert isinstance(es, EarlyStopping)
        assert es.patience == 5

    def test_create_custom(self):
        """Test custom creation."""
        es = create_early_stopping(patience=10, min_delta=0.01, mode='max')
        assert es.patience == 10
        assert es.min_delta == 0.01
        assert es.mode == 'max'


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestValidateEarlyStopping:
    """Tests for validate_early_stopping utility."""

    def test_validate_loss(self, simple_model, simple_dataloader):
        """Test validation loss calculation."""
        criterion = nn.MSELoss()
        loss = validate_early_stopping(
            simple_model,
            simple_dataloader,
            criterion,
            device='cpu',
        )

        assert isinstance(loss, float)
        assert loss > 0  # Should have some loss

    def test_validate_empty_loader(self, simple_model):
        """Test validation with empty loader."""
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randn(0, 1))
        empty_loader = DataLoader(empty_dataset, batch_size=10)
        criterion = nn.MSELoss()

        loss = validate_early_stopping(
            simple_model,
            empty_loader,
            criterion,
            device='cpu',
        )

        assert loss == 0.0


# =============================================================================
# Test EarlyStoppingTrainer
# =============================================================================


class TestEarlyStoppingTrainer:
    """Tests for EarlyStoppingTrainer."""

    def test_init(self, simple_model):
        """Test trainer initialization."""
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = EarlyStoppingTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            patience=5,
            device='cpu',
        )

        assert trainer.early_stopping.patience == 5
        assert trainer.early_stopping is not None

    def test_train_epoch(self, simple_model, simple_dataloader):
        """Test single epoch training."""
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = EarlyStoppingTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
        )

        loss = trainer.train_epoch(simple_dataloader)

        assert isinstance(loss, float)
        assert loss > 0

    def test_validate(self, simple_model, simple_dataloader):
        """Test validation."""
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = EarlyStoppingTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
        )

        loss = trainer.validate(simple_dataloader)

        assert isinstance(loss, float)

    def test_full_training(self, simple_model, simple_dataloader):
        """Test full training with early stopping."""
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer = EarlyStoppingTrainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            patience=3,
            device='cpu',
        )

        history = trainer.train(
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            epochs=20,
            verbose=False,
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'stopped_early' in history
        assert 'best_epoch' in history
        assert len(history['train_loss']) > 0


# =============================================================================
# Test Registry
# =============================================================================


class TestRegistry:
    """Tests for component registry."""

    def test_early_stopping_components(self):
        """Test early stopping components registry."""
        assert isinstance(EARLY_STOPPING_COMPONENTS, dict)
        assert 'callback' in EARLY_STOPPING_COMPONENTS
        assert 'trainer' in EARLY_STOPPING_COMPONENTS


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for early stopping."""

    def test_typical_training_workflow(self, simple_model, simple_dataloader):
        """Test typical early stopping workflow."""
        es = EarlyStopping(patience=5, restore_best_weights=True, verbose=False)
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        initial_weights = simple_model.weight.clone()

        for epoch in range(100):
            # Train
            simple_model.train()
            for inputs, targets in simple_dataloader:
                optimizer.zero_grad()
                outputs = simple_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validate
            val_loss = validate_early_stopping(
                simple_model, simple_dataloader, criterion, 'cpu'
            )

            # Early stopping check
            if es(val_loss, simple_model, epoch + 1):
                break

        # Restore best weights
        es.restore_weights(simple_model)

        # Should have stopped before 100 epochs or at end
        assert es.best_epoch > 0

    def test_mode_max_accuracy(self):
        """Test mode='max' for accuracy monitoring."""
        es = EarlyStopping(patience=5, mode='max', verbose=False)

        # Improving accuracy
        es.step(0.7)  # Initial
        es.step(0.8)  # Better
        es.step(0.85)  # Better

        assert es.counter == 0

        # Declining
        es.step(0.8)
        es.step(0.75)

        assert es.counter == 2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
