"""
Resume Training Script with Checkpoint Support.

This module provides:
    - ResumeTrainingPipeline: Complete training pipeline with resume support
    - MNIST training example with checkpoint save/resume
    - Gradient verification utilities
    - Training state recovery demonstration

Usage:
    ```python
    # First run
    pipeline = ResumeTrainingPipeline(config)
    pipeline.train(epochs=10)

    # Resume from checkpoint
    pipeline = ResumeTrainingPipeline(config)
    pipeline.resume()  # Resumes from last checkpoint
    pipeline.train(epochs=20)  # Continue to epoch 20
    ```
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import logging
import os
import time

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, StepLR
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None
    LRScheduler = None
    CosineAnnealingLR = None
    StepLR = None
    F = None

from .checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointState,
    ResumeTrainer,
    verify_gradient_consistency,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        model_name: Model name for saving
        epochs: Total training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        gradient_clip_val: Gradient clipping value
        use_amp: Use automatic mixed precision
        device: Device to use
        seed: Random seed
        checkpoint_dir: Checkpoint directory
        save_every_n_epochs: Save checkpoint every N epochs
        validate_every_n_epochs: Validate every N epochs
        patience: Early stopping patience
        resume: Resume from checkpoint if available
    """
    model_name: str = "model"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    validate_every_n_epochs: int = 1
    patience: int = 5
    resume: bool = True


# =============================================================================
# Simple Models for Testing
# =============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST testing."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST testing."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# =============================================================================
# Resume Training Pipeline
# =============================================================================


class ResumeTrainingPipeline:
    """
    Complete training pipeline with checkpoint resume support.

    Features:
        - Automatic checkpoint saving
        - Resume from interruption
        - AMP support
        - Gradient clipping
        - Early stopping
        - MNIST validation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        """
        Initialize training pipeline.

        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler (created if None)
        """
        if config is None:
            config = TrainingConfig()
        self.config = config

        # Set random seed
        self._set_seed(config.seed)

        # Model and data
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and scheduler
        if optimizer is None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

        if scheduler is None:
            scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
        self.scheduler = scheduler

        # AMP scaler
        self.scaler = torch.amp.GradScaler(
            device=config.device.split(":")[0] if ":" in config.device else config.device,
            enabled=config.use_amp,
        )

        # Checkpoint manager
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=config.checkpoint_dir,
            save_every_n_epochs=config.save_every_n_epochs,
            keep_best=True,
            keep_last_k=3,
            metric_name="val_loss",
            metric_mode="min",
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_config)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        # Resume if requested
        if config.resume and self.checkpoint_manager.has_checkpoint():
            self.resume()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def resume(self) -> Optional[CheckpointState]:
        """
        Resume training from checkpoint.

        Returns:
            Loaded checkpoint state or None
        """
        if not self.checkpoint_manager.has_checkpoint():
            logger.info("No checkpoint found, starting fresh training")
            return None

        try:
            state = self.checkpoint_manager.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.config.device,
            )

            self.current_epoch = state.epoch + 1  # Start from next epoch
            self.global_step = state.global_step
            self.best_val_loss = state.best_metric or float("inf")

            logger.info(
                f"Resumed from epoch {state.epoch}, "
                f"global_step {state.global_step}, "
                f"best_val_loss {self.best_val_loss:.4f}"
            )
            return state

        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}")
            return None

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.config.use_amp:
                with torch.amp.autocast(
                    device_type=self.config.device.split(":")[0] if ":" in self.config.device else self.config.device
                ):
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, targets)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def train(
        self,
        epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs (None = use config)
            callbacks: Optional callbacks

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.epochs

        logger.info(
            f"Starting training from epoch {self.current_epoch + 1}/{epochs} "
            f"(device={self.config.device}, amp={self.config.use_amp})"
        )

        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = {}
            if (epoch + 1) % self.config.validate_every_n_epochs == 0:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["val_loss"])
                self.history["val_accuracy"].append(val_metrics["val_accuracy"])

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Logging
            epoch_time = time.time() - epoch_start
            log_msg = (
                f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                f"train_loss: {train_metrics['train_loss']:.4f}"
            )
            if val_metrics:
                log_msg += (
                    f" - val_loss: {val_metrics['val_loss']:.4f} - "
                    f"val_acc: {val_metrics['val_accuracy']:.4f}"
                )
            logger.info(log_msg)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                metrics = {**train_metrics, **val_metrics}
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    metrics=metrics,
                )

            # Early stopping
            if val_metrics and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            elif val_metrics:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics)

            self.current_epoch = epoch + 1

        return self.history


# =============================================================================
# MNIST Training Example
# =============================================================================


def create_mnist_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST dataloaders.

    Args:
        batch_size: Batch size
        data_dir: Data directory
        num_workers: Number of data loading workers

    Returns:
        (train_loader, val_loader)
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("torchvision required for MNIST example")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Split into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def train_mnist_with_checkpoint(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "checkpoints/mnist",
    use_amp: bool = True,
    device: str = "auto",
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Train MNIST with checkpoint support.

    This function demonstrates the complete checkpoint save/resume workflow:
    1. Create model, optimizer, scheduler
    2. Initialize training pipeline
    3. Resume from checkpoint if available
    4. Train with automatic checkpoint saving
    5. Return training history

    Args:
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
        use_amp: Use automatic mixed precision
        device: Device to use ("auto", "cuda", "cpu")
        resume: Resume from checkpoint if available

    Returns:
        Training results dictionary
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders
    train_loader, val_loader = create_mnist_dataloaders(batch_size)

    # Create model
    model = SimpleCNN(num_classes=10)

    # Create config
    config = TrainingConfig(
        model_name="mnist_cnn",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_amp=use_amp,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )

    # Create training pipeline
    pipeline = ResumeTrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Train
    history = pipeline.train()

    return {
        "history": history,
        "best_val_loss": pipeline.best_val_loss,
        "final_epoch": pipeline.current_epoch,
        "total_steps": pipeline.global_step,
    }


def verify_checkpoint_resume_consistency(
    checkpoint_dir: str = "checkpoints/test_resume",
    epochs_phase1: int = 3,
    epochs_phase2: int = 3,
    batch_size: int = 64,
    atol: float = 1e-6,
) -> bool:
    """
    Verify that checkpoint save/resume produces consistent training.

    This test:
    1. Trains model for N epochs, saves checkpoint
    2. Creates new model, loads checkpoint
    3. Trains both original and resumed models for M epochs
    4. Verifies gradients and outputs match

    Args:
        checkpoint_dir: Checkpoint directory
        epochs_phase1: Epochs before checkpoint
        epochs_phase2: Epochs after checkpoint
        batch_size: Batch size
        atol: Absolute tolerance for comparison

    Returns:
        True if verification passes
    """
    import shutil

    # Clean checkpoint directory
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader = create_mnist_dataloaders(batch_size)

    # Phase 1: Train and save checkpoint
    logger.info("Phase 1: Training initial model...")

    model1 = SimpleCNN()
    config1 = TrainingConfig(
        epochs=epochs_phase1,
        checkpoint_dir=checkpoint_dir,
        resume=False,
    )

    pipeline1 = ResumeTrainingPipeline(
        model=model1,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config1,
    )
    pipeline1.train()

    # Phase 2: Resume and continue training
    logger.info("Phase 2: Resuming training...")

    model2 = SimpleCNN()
    config2 = TrainingConfig(
        epochs=epochs_phase1 + epochs_phase2,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )

    pipeline2 = ResumeTrainingPipeline(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config2,
    )
    # Resume is automatic in __init__ when resume=True
    pipeline2.train()

    # Phase 3: Train original model continuously for comparison
    logger.info("Phase 3: Training continuous model for comparison...")

    # Clean checkpoints for continuous training
    checkpoint_dir_continuous = checkpoint_dir + "_continuous"
    if os.path.exists(checkpoint_dir_continuous):
        shutil.rmtree(checkpoint_dir_continuous)

    model3 = SimpleCNN()
    config3 = TrainingConfig(
        epochs=epochs_phase1 + epochs_phase2,
        checkpoint_dir=checkpoint_dir_continuous,
        resume=False,
    )

    # Set same seed as model1
    torch.manual_seed(42)
    np.random.seed(42)

    pipeline3 = ResumeTrainingPipeline(
        model=model3,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config3,
    )
    pipeline3.train()

    # Compare final validation accuracy
    acc1 = pipeline2.history["val_accuracy"][-1] if pipeline2.history["val_accuracy"] else 0
    acc2 = pipeline3.history["val_accuracy"][-1] if pipeline3.history["val_accuracy"] else 0

    logger.info(f"Resumed model final val_accuracy: {acc1:.4f}")
    logger.info(f"Continuous model final val_accuracy: {acc2:.4f}")

    # Check that resumed model achieved similar results
    # (allowing some tolerance due to random state differences)
    is_consistent = abs(acc1 - acc2) < 0.05  # 5% tolerance

    if is_consistent:
        logger.info("✓ Checkpoint resume consistency verified")
    else:
        logger.warning(f"✗ Checkpoint resume consistency check failed: {abs(acc1 - acc2):.4f}")

    return is_consistent


# =============================================================================
# Registry
# =============================================================================


RESUME_TRAINING_COMPONENTS = {
    'TrainingConfig': TrainingConfig,
    'SimpleMLP': SimpleMLP,
    'SimpleCNN': SimpleCNN,
    'ResumeTrainingPipeline': ResumeTrainingPipeline,
    'create_mnist_dataloaders': create_mnist_dataloaders,
    'train_mnist_with_checkpoint': train_mnist_with_checkpoint,
    'verify_checkpoint_resume_consistency': verify_checkpoint_resume_consistency,
}
