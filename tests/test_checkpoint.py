"""
Unit tests for Checkpoint Manager and Resume Training.

Tests:
    - CheckpointManager save/load functionality
    - AMP scaler state persistence
    - Gradient consistency verification
    - Training resume functionality
    - MNIST training with checkpoint
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from phase5_deployment.checkpoint_manager import (
    CheckpointPolicy,
    CheckpointConfig,
    CheckpointState,
    CheckpointManager,
    ResumeTrainer,
    create_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
    verify_gradient_consistency,
)
from phase5_deployment.resume_training import (
    TrainingConfig,
    SimpleMLP,
    SimpleCNN,
    ResumeTrainingPipeline,
    create_mnist_dataloaders,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)


@pytest.fixture
def simple_optimizer(simple_model):
    """Create simple optimizer for testing."""
    return optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def simple_scheduler(simple_optimizer):
    """Create simple scheduler for testing."""
    return optim.lr_scheduler.StepLR(simple_optimizer, step_size=5, gamma=0.1)


@pytest.fixture
def amp_scaler():
    """Create AMP scaler for testing."""
    return torch.amp.GradScaler(device="cpu", enabled=True)


# =============================================================================
# CheckpointConfig Tests
# =============================================================================


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointConfig()

        assert config.checkpoint_dir == "checkpoints"
        assert config.save_every_n_epochs == 1
        assert config.keep_best is True
        assert config.keep_last_k == 3
        assert config.policy == CheckpointPolicy.LATEST_ONLY
        assert config.metric_name == "val_loss"
        assert config.metric_mode == "min"

    def test_custom_config(self, temp_checkpoint_dir):
        """Test custom configuration values."""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best=False,
            keep_last_k=5,
            metric_mode="max",
        )

        assert config.checkpoint_dir == temp_checkpoint_dir
        assert config.keep_best is False
        assert config.keep_last_k == 5
        assert config.metric_mode == "max"

    def test_invalid_metric_mode(self):
        """Test that invalid metric mode raises error."""
        with pytest.raises(ValueError):
            CheckpointConfig(metric_mode="invalid")

    def test_negative_keep_last_k(self):
        """Test that negative keep_last_k raises error."""
        with pytest.raises(ValueError):
            CheckpointConfig(keep_last_k=-1)


# =============================================================================
# CheckpointState Tests
# =============================================================================


class TestCheckpointState:
    """Tests for CheckpointState."""

    def test_to_dict_and_from_dict(self, simple_model, simple_optimizer):
        """Test serialization and deserialization."""
        state = CheckpointState(
            epoch=5,
            global_step=1000,
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict=simple_optimizer.state_dict(),
            best_metric=0.5,
            metrics={"val_loss": 0.5, "val_acc": 0.85},
        )

        # Convert to dict and back
        state_dict = state.to_dict()
        restored = CheckpointState.from_dict(state_dict)

        assert restored.epoch == state.epoch
        assert restored.global_step == state.global_step
        assert restored.best_metric == state.best_metric
        assert restored.metrics == state.metrics

    def test_optional_fields(self, simple_model):
        """Test that optional fields can be None."""
        state = CheckpointState(
            epoch=0,
            global_step=0,
            model_state_dict=simple_model.state_dict(),
        )

        assert state.optimizer_state_dict is None
        assert state.scheduler_state_dict is None
        assert state.scaler_state_dict is None
        assert state.best_metric is None


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_initialization(self, temp_checkpoint_dir):
        """Test checkpoint manager initialization."""
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)

        assert manager.config == config
        assert Path(temp_checkpoint_dir).exists()

    def test_save_checkpoint(
        self, simple_model, simple_optimizer, simple_scheduler, amp_scaler, temp_checkpoint_dir
    ):
        """Test saving checkpoint."""
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)

        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=simple_scheduler,
            scaler=amp_scaler,
            epoch=5,
            global_step=100,
            metrics={"val_loss": 0.5},
        )

        assert checkpoint_path.exists()
        assert manager.latest_checkpoint_path.exists()

    def test_load_checkpoint(
        self, simple_model, simple_optimizer, simple_scheduler, amp_scaler, temp_checkpoint_dir
    ):
        """Test loading checkpoint."""
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)

        # Save
        manager.save(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=simple_scheduler,
            scaler=amp_scaler,
            epoch=5,
            global_step=100,
            metrics={"val_loss": 0.5},
        )

        # Create new model/optimizer
        new_model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
        new_scaler = torch.amp.GradScaler(device="cpu", enabled=True)

        # Load
        state = manager.load(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=new_scaler,
        )

        assert state.epoch == 5
        assert state.global_step == 100
        assert state.metrics["val_loss"] == 0.5

    def test_save_and_load_best_model(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test best model saving and loading."""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best=True,
            metric_name="val_loss",
            metric_mode="min",
        )
        manager = CheckpointManager(config)

        # Save with loss=0.5 (first, should be best)
        manager.save(
            model=simple_model,
            optimizer=simple_optimizer,
            epoch=1,
            metrics={"val_loss": 0.5},
        )
        # First save with metric should create best model
        assert manager.best_checkpoint_path.exists()

        # Save with loss=0.3 (better, should update best)
        manager.save(
            model=simple_model,
            optimizer=simple_optimizer,
            epoch=2,
            metrics={"val_loss": 0.3},
        )

        # Save with loss=0.4 (worse, should not update best)
        manager.save(
            model=simple_model,
            optimizer=simple_optimizer,
            epoch=3,
            metrics={"val_loss": 0.4},
        )

        # Load best model
        new_model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        state = manager.load(model=new_model, load_best=True)

        # Best should be from epoch 2 (loss=0.3)
        assert state.epoch == 2
        assert manager._best_metric == 0.3

    def test_checkpoint_cleanup(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test old checkpoint cleanup."""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            keep_last_k=2,
        )
        manager = CheckpointManager(config)

        # Save 5 checkpoints
        for i in range(5):
            manager.save(
                model=simple_model,
                optimizer=simple_optimizer,
                epoch=i,
                global_step=i * 100,
            )

        # Should only have 2 epoch checkpoints + latest + best
        checkpoint_files = list(Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pth"))
        assert len(checkpoint_files) <= 2

    def test_has_checkpoint(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test has_checkpoint method."""
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)

        assert not manager.has_checkpoint()

        manager.save(model=simple_model, optimizer=simple_optimizer, epoch=1)

        assert manager.has_checkpoint()

    def test_delete_all(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test delete_all method."""
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)

        manager.save(model=simple_model, optimizer=simple_optimizer, epoch=1)
        assert manager.has_checkpoint()

        manager.delete_all()
        assert not manager.has_checkpoint()


# =============================================================================
# ResumeTrainer Tests
# =============================================================================


class TestResumeTrainer:
    """Tests for ResumeTrainer."""

    def test_initialization(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test resume trainer initialization."""
        checkpoint_config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = ResumeTrainer(
            model=simple_model,
            optimizer=simple_optimizer,
            checkpoint_manager=CheckpointManager(checkpoint_config),
        )

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_metric is None

    def test_save_and_resume(
        self, simple_model, simple_optimizer, simple_scheduler, amp_scaler, temp_checkpoint_dir
    ):
        """Test save and resume functionality."""
        checkpoint_config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(checkpoint_config)

        # Create first trainer
        trainer1 = ResumeTrainer(
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=simple_scheduler,
            scaler=amp_scaler,
            checkpoint_manager=manager,
        )

        # Update state
        trainer1._current_epoch = 5
        trainer1._global_step = 500
        trainer1._best_metric = 0.5

        # Save
        trainer1.save_checkpoint(metrics={"val_loss": 0.5})

        # Create new trainer to resume
        new_model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
        new_scaler = torch.amp.GradScaler(device="cpu", enabled=True)

        trainer2 = ResumeTrainer(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=new_scaler,
            checkpoint_manager=manager,
        )

        # Resume
        state = trainer2.load_checkpoint()

        assert trainer2.current_epoch == 5
        assert trainer2.global_step == 500
        assert trainer2.best_metric == 0.5

    def test_train_step(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test train step execution."""
        checkpoint_config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        trainer = ResumeTrainer(
            model=simple_model,
            optimizer=simple_optimizer,
            checkpoint_manager=CheckpointManager(checkpoint_config),
        )

        # Create dummy batch
        inputs = torch.randn(32, 784)
        targets = torch.randint(0, 10, (32,))
        batch = (inputs, targets)

        # Train step
        metrics = trainer.train_step(
            batch=batch,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
        )

        assert "loss" in metrics
        assert trainer.global_step == 1


# =============================================================================
# Gradient Consistency Tests
# =============================================================================


class TestGradientConsistency:
    """Tests for gradient consistency verification."""

    def test_identical_models(self, simple_model):
        """Test that identical models pass verification."""
        model1 = simple_model
        model2 = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)

        # Copy weights
        model2.load_state_dict(model1.state_dict())

        # Without gradients, just check parameters match
        assert verify_gradient_consistency(model1, model2, check_gradients=False)

    def test_different_models(self, simple_model):
        """Test that different models fail verification."""
        model1 = simple_model
        model2 = SimpleMLP(input_size=784, hidden_sizes=[256, 128], num_classes=10)

        # Should fail due to different architectures
        assert not verify_gradient_consistency(model1, model2)

    def test_optimizer_state_consistency(self, temp_checkpoint_dir):
        """Test optimizer state consistency after checkpoint save/load."""
        # Create model and optimizer
        model1 = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

        # Run a few steps to build optimizer state
        inputs = torch.randn(32, 784)
        targets = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()

        for _ in range(3):
            optimizer1.zero_grad()
            loss = criterion(model1(inputs), targets)
            loss.backward()
            optimizer1.step()

        # Save checkpoint
        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
        manager = CheckpointManager(config)
        manager.save(model=model1, optimizer=optimizer1, epoch=3)

        # Load into new model
        model2 = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        manager.load(model=model2, optimizer=optimizer2)

        # Verify consistency after checkpoint load
        assert verify_gradient_consistency(model1, model2, optimizer1, optimizer2, check_gradients=False)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_checkpoint_manager(self, temp_checkpoint_dir):
        """Test factory function."""
        manager = create_checkpoint_manager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best=False,
            keep_last_k=5,
        )

        assert isinstance(manager, CheckpointManager)
        assert manager.config.checkpoint_dir == temp_checkpoint_dir
        assert manager.config.keep_best is False
        assert manager.config.keep_last_k == 5

    def test_save_load_checkpoint_convenience(self, simple_model, simple_optimizer, temp_checkpoint_dir):
        """Test convenience save/load functions."""
        path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pth")

        # Save
        save_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            path=path,
            epoch=5,
            global_step=100,
            metrics={"val_loss": 0.5},
        )

        assert os.path.exists(path)

        # Load
        new_model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

        checkpoint = load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            path=path,
        )

        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["metrics"]["val_loss"] == 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for checkpoint functionality."""

    def test_full_training_cycle(self, temp_checkpoint_dir):
        """Test full training cycle with checkpoint save/resume."""
        # Create simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000, seed=42):
                torch.manual_seed(seed)
                self.data = torch.randn(size, 784)
                self.targets = torch.randint(0, 10, (size,))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        train_dataset = SimpleDataset(1000)
        val_dataset = SimpleDataset(200)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        # Phase 1: Train for 3 epochs
        torch.manual_seed(42)  # Set seed for reproducibility
        model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        config = TrainingConfig(
            epochs=3,
            checkpoint_dir=temp_checkpoint_dir,
            save_every_n_epochs=1,
            validate_every_n_epochs=1,
            device="cpu",
            use_amp=False,
            resume=False,
        )

        pipeline1 = ResumeTrainingPipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        history1 = pipeline1.train()

        assert len(history1["train_loss"]) == 3
        final_loss_phase1 = history1["train_loss"][-1]

        # Phase 2: Resume and train 2 more epochs
        model2 = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10)
        config2 = TrainingConfig(
            epochs=5,
            checkpoint_dir=temp_checkpoint_dir,
            save_every_n_epochs=1,
            validate_every_n_epochs=1,
            device="cpu",
            use_amp=False,
            resume=True,
        )

        pipeline2 = ResumeTrainingPipeline(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config2,
        )

        # Check that we resumed from epoch 3
        assert pipeline2.current_epoch == 3

        # Continue training
        history2 = pipeline2.train()

        # Should have 2 new entries (epochs 4 and 5)
        assert len(history2["train_loss"]) == 2

        # Loss should continue from where it left off (roughly)
        # Not a strict assertion as training is stochastic
        assert history2["train_loss"][0] < 2.0  # Reasonable loss value

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_amp_state_preservation(self, temp_checkpoint_dir):
        """Test that AMP scaler state is preserved."""
        model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)

        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir, save_scaler=True)
        manager = CheckpointManager(config)

        # Run a training step to initialize scaler
        inputs = torch.randn(32, 784).cuda()
        targets = torch.randint(0, 10, (32,)).cuda()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Get initial scale
        initial_scale = scaler.get_scale()

        # Save checkpoint
        manager.save(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=1,
        )

        # Create new scaler and load
        new_model = SimpleMLP(input_size=784, hidden_sizes=[128], num_classes=10).cuda()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scaler = torch.amp.GradScaler(device="cuda", enabled=True)

        manager.load(
            model=new_model,
            optimizer=new_optimizer,
            scaler=new_scaler,
        )

        # Verify scaler state was restored
        assert new_scaler.get_scale() == initial_scale


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
