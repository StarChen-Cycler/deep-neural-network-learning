"""
Checkpoint Manager for Training State Persistence.

This module provides:
    - CheckpointManager: Save/load training state
    - Best model tracking and automatic rotation
    - AMP GradScaler state support
    - Gradient state preservation
    - Resume training utilities

Theory:
    Checkpointing allows training to be resumed after interruption.
    A complete checkpoint includes:
        - Model weights (state_dict)
        - Optimizer state (momentum, learning rate history)
        - Scheduler state (learning rate schedule position)
        - AMP scaler state (loss scale for mixed precision)
        - Training metadata (epoch, step, best metric)
        - Random state (for reproducibility)

References:
    - PyTorch Saving/Loading: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    - AMP Checkpointing: https://pytorch.org/docs/stable/amp.html
"""

from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import copy
import json
import time
import logging
import os
import random
import shutil

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    Optimizer = None
    LRScheduler = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class CheckpointPolicy(Enum):
    """Checkpoint saving policies."""
    BEST_ONLY = "best_only"  # Only save best model
    LATEST_ONLY = "latest_only"  # Only save latest checkpoint
    ALL = "all"  # Save all checkpoints
    TOP_K = "top_k"  # Keep top K best checkpoints


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint management.

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        save_every_n_epochs: Save every N epochs (0 = disabled)
        save_every_n_steps: Save every N steps (0 = disabled)
        keep_best: Keep the best model separately
        keep_last_k: Keep last K checkpoints (0 = all)
        policy: Checkpoint saving policy
        metric_name: Metric to track for best model
        metric_mode: 'min' or 'max' for best metric
        save_optimizer: Save optimizer state
        save_scheduler: Save scheduler state
        save_scaler: Save AMP scaler state
        save_random_state: Save random state for reproducibility
    """
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 0
    keep_best: bool = True
    keep_last_k: int = 3
    policy: CheckpointPolicy = CheckpointPolicy.LATEST_ONLY
    metric_name: str = "val_loss"
    metric_mode: str = "min"
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_scaler: bool = True
    save_random_state: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.metric_mode not in ("min", "max"):
            raise ValueError(f"metric_mode must be 'min' or 'max', got {self.metric_mode}")
        if self.keep_last_k < 0:
            raise ValueError(f"keep_last_k must be >= 0, got {self.keep_last_k}")


# =============================================================================
# Checkpoint Data Structures
# =============================================================================


@dataclass
class CheckpointState:
    """
    Container for checkpoint state.

    Attributes:
        epoch: Current epoch
        global_step: Global training step
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        scheduler_state_dict: Scheduler state dictionary
        scaler_state_dict: AMP scaler state dictionary
        best_metric: Best metric value
        metrics: Additional metrics
        config: Training configuration
        random_state: Random state for reproducibility
        timestamp: Checkpoint creation time
        version: Checkpoint format version
    """
    epoch: int
    global_step: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    scaler_state_dict: Optional[Dict[str, Any]] = None
    best_metric: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    random_state: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H-%M-%S"))
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "scaler_state_dict": self.scaler_state_dict,
            "best_metric": self.best_metric,
            "metrics": self.metrics,
            "config": self.config,
            "random_state": self.random_state,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        return cls(
            epoch=data["epoch"],
            global_step=data["global_step"],
            model_state_dict=data["model_state_dict"],
            optimizer_state_dict=data.get("optimizer_state_dict"),
            scheduler_state_dict=data.get("scheduler_state_dict"),
            scaler_state_dict=data.get("scaler_state_dict"),
            best_metric=data.get("best_metric"),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            random_state=data.get("random_state"),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", "1.0"),
        )


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """
    Manages training checkpoints with support for:
        - Model, optimizer, scheduler, AMP scaler states
        - Best model tracking
        - Automatic checkpoint rotation
        - Gradient state preservation

    Usage:
        ```python
        manager = CheckpointManager(config)

        # During training
        manager.save(
            model, optimizer, scheduler, scaler,
            epoch=5, global_step=1000, metrics={"val_loss": 0.5}
        )

        # Resume training
        state = manager.load(model, optimizer, scheduler, scaler)
        start_epoch = state.epoch + 1
        ```
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        if config is None:
            config = CheckpointConfig()
        self.config = config
        self._best_metric: Optional[float] = None
        self._checkpoint_history: List[str] = []

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_checkpoint_path(self) -> Path:
        """Path to best checkpoint."""
        return self.checkpoint_dir / "best_model.pth"

    @property
    def latest_checkpoint_path(self) -> Path:
        """Path to latest checkpoint."""
        return self.checkpoint_dir / "latest_checkpoint.pth"

    def _get_random_state(self) -> Optional[Dict[str, Any]]:
        """Get random state for reproducibility."""
        if not self.config.save_random_state:
            return None

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

        if HAS_TORCH:
            state["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()

        return state

    def _set_random_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Set random state for reproducibility."""
        if state is None:
            return

        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])

        if HAS_TORCH:
            if "torch" in state:
                torch.set_rng_state(state["torch"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _is_better_metric(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self._best_metric is None:
            return True

        if self.config.metric_mode == "min":
            return metric < self._best_metric
        else:
            return metric > self._best_metric

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on keep_last_k policy."""
        if self.config.keep_last_k == 0:
            return

        # Get all checkpoint files (excluding best_model.pth)
        checkpoints = [
            f for f in self.checkpoint_dir.glob("checkpoint_epoch_*.pth")
        ]
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for old_checkpoint in checkpoints[self.config.keep_last_k:]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not remove {old_checkpoint}: {e}")

    def save(
        self,
        model: "nn.Module",
        optimizer: Optional["Optimizer"] = None,
        scheduler: Optional["LRScheduler"] = None,
        scaler: Optional["torch.amp.GradScaler"] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: AMP gradient scaler
            epoch: Current epoch
            global_step: Global training step
            metrics: Current metrics
            config: Training configuration

        Returns:
            Path to saved checkpoint
        """
        if metrics is None:
            metrics = {}

        # Get state dictionaries
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict() if optimizer and self.config.save_optimizer else None
        scheduler_state = scheduler.state_dict() if scheduler and self.config.save_scheduler else None
        scaler_state = scaler.state_dict() if scaler and self.config.save_scaler else None

        # Get current metric
        current_metric = metrics.get(self.config.metric_name) if metrics else None
        is_best = current_metric is not None and self._is_better_metric(current_metric)

        # Update best metric if this is the best
        if is_best:
            self._best_metric = current_metric

        # Create checkpoint state
        checkpoint = CheckpointState(
            epoch=epoch,
            global_step=global_step,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            scheduler_state_dict=scheduler_state,
            scaler_state_dict=scaler_state,
            best_metric=self._best_metric,
            metrics=metrics or {},
            config=config or {},
            random_state=self._get_random_state(),
        )

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint.to_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update latest symlink/copy
        shutil.copy(checkpoint_path, self.latest_checkpoint_path)

        # Save best model if applicable
        if self.config.keep_best and is_best:
            shutil.copy(checkpoint_path, self.best_checkpoint_path)
            logger.info(f"Saved best model (metric={current_metric:.4f})")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        model: "nn.Module",
        optimizer: Optional["Optimizer"] = None,
        scheduler: Optional["LRScheduler"] = None,
        scaler: Optional["torch.amp.GradScaler"] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        device: Optional[str] = None,
    ) -> CheckpointState:
        """
        Load checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: AMP gradient scaler
            checkpoint_path: Specific checkpoint to load (None = latest)
            load_best: Load best model instead of latest
            device: Device to load to

        Returns:
            Loaded checkpoint state
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.best_checkpoint_path
            else:
                checkpoint_path = self.latest_checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint = CheckpointState.from_dict(checkpoint_data)

        # Load model state
        model.load_state_dict(checkpoint.model_state_dict)

        # Load optimizer state
        if optimizer and checkpoint.optimizer_state_dict:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        # Load scheduler state
        if scheduler and checkpoint.scheduler_state_dict:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)

        # Load scaler state
        if scaler and checkpoint.scaler_state_dict:
            scaler.load_state_dict(checkpoint.scaler_state_dict)

        # Restore random state
        self._set_random_state(checkpoint.random_state)

        # Update best metric
        self._best_metric = checkpoint.best_metric

        logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch={checkpoint.epoch})")
        return checkpoint

    def get_available_checkpoints(self) -> List[Path]:
        """Get list of available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        return checkpoints

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return self.latest_checkpoint_path.exists() or self.best_checkpoint_path.exists()

    def delete_all(self) -> None:
        """Delete all checkpoints."""
        for checkpoint in self.checkpoint_dir.glob("*.pth"):
            checkpoint.unlink()
        logger.info("Deleted all checkpoints")


# =============================================================================
# Resume Training Utilities
# =============================================================================


class ResumeTrainer:
    """
    Trainer with checkpoint resume support.

    Provides:
        - Seamless training continuation
        - Automatic checkpoint saving
        - Training state recovery
    """

    def __init__(
        self,
        model: "nn.Module",
        optimizer: "Optimizer",
        scheduler: Optional["LRScheduler"] = None,
        scaler: Optional["torch.amp.GradScaler"] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        """
        Initialize resume trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: AMP gradient scaler
            checkpoint_manager: Checkpoint manager
            checkpoint_config: Checkpoint config (if no manager)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(checkpoint_config)
        self.checkpoint_manager = checkpoint_manager

        self._current_epoch = 0
        self._global_step = 0
        self._best_metric: Optional[float] = None

    @property
    def current_epoch(self) -> int:
        """Current epoch."""
        return self._current_epoch

    @property
    def global_step(self) -> int:
        """Global step."""
        return self._global_step

    @property
    def best_metric(self) -> Optional[float]:
        """Best metric."""
        return self._best_metric

    def save_checkpoint(
        self,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save current training state.

        Args:
            metrics: Current metrics
            config: Training configuration

        Returns:
            Path to saved checkpoint
        """
        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self._current_epoch,
            global_step=self._global_step,
            metrics=metrics,
            config=config,
        )

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        device: Optional[str] = None,
    ) -> CheckpointState:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load best model
            device: Device to load to

        Returns:
            Loaded checkpoint state
        """
        state = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            checkpoint_path=checkpoint_path,
            load_best=load_best,
            device=device,
        )

        self._current_epoch = state.epoch
        self._global_step = state.global_step
        self._best_metric = state.best_metric

        return state

    def can_resume(self) -> bool:
        """Check if resume is possible."""
        return self.checkpoint_manager.has_checkpoint()

    def train_step(
        self,
        batch: Any,
        criterion: Callable,
        device: str = "cpu",
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Input batch (inputs, targets)
            criterion: Loss function
            device: Device to use
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Step metrics
        """
        self.model.train()

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        self.optimizer.zero_grad()

        if use_amp and self.scaler is not None:
            with torch.amp.autocast(device_type=device.split(":")[0] if ":" in device else device):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets) / gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (self._global_step + 1) % gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets) / gradient_accumulation_steps
            loss.backward()

            if (self._global_step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()

        self._global_step += 1

        return {"loss": loss.item() * gradient_accumulation_steps}

    def validate(
        self,
        dataloader: Any,
        criterion: Callable,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            dataloader: Validation dataloader
            criterion: Loss function
            device: Device to use

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return {
            "val_loss": total_loss / len(dataloader),
            "val_accuracy": correct / total if total > 0 else 0.0,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    keep_best: bool = True,
    keep_last_k: int = 3,
    **kwargs
) -> CheckpointManager:
    """
    Factory function to create checkpoint manager.

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_best: Keep best model
        keep_last_k: Keep last K checkpoints
        **kwargs: Additional configuration options

    Returns:
        CheckpointManager instance
    """
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        keep_best=keep_best,
        keep_last_k=keep_last_k,
        **kwargs
    )
    return CheckpointManager(config)


def save_checkpoint(
    model: "nn.Module",
    optimizer: Optional["Optimizer"] = None,
    scheduler: Optional["LRScheduler"] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    path: str = "checkpoint.pth",
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Convenience function to save checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Scheduler
        scaler: AMP scaler
        path: Save path
        epoch: Current epoch
        global_step: Global step
        metrics: Metrics to save
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: "nn.Module",
    optimizer: Optional["Optimizer"] = None,
    scheduler: Optional["LRScheduler"] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    path: str = "checkpoint.pth",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Scheduler
        scaler: AMP scaler
        path: Checkpoint path
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint


def verify_gradient_consistency(
    model1: "nn.Module",
    model2: "nn.Module",
    optimizer1: Optional["Optimizer"] = None,
    optimizer2: Optional["Optimizer"] = None,
    atol: float = 1e-6,
    check_gradients: bool = False,
) -> bool:
    """
    Verify gradient consistency between two model states.

    Used to ensure checkpoint save/load preserves gradient state.

    Args:
        model1: First model
        model2: Second model
        optimizer1: First optimizer
        optimizer2: Second optimizer
        atol: Absolute tolerance
        check_gradients: Whether to check gradients (requires backward pass)

    Returns:
        True if parameters are consistent
    """
    # Get parameter lists
    params1 = list(model1.named_parameters())
    params2 = list(model2.named_parameters())

    # Check same number of parameters
    if len(params1) != len(params2):
        logger.warning(f"Parameter count mismatch: {len(params1)} vs {len(params2)}")
        return False

    # Check model parameters
    for (name1, param1), (name2, param2) in zip(params1, params2):
        if name1 != name2:
            logger.warning(f"Parameter name mismatch: {name1} vs {name2}")
            return False

        # Check shapes match
        if param1.shape != param2.shape:
            logger.warning(f"Parameter shape mismatch for {name1}: {param1.shape} vs {param2.shape}")
            return False

        if not torch.allclose(param1.data, param2.data, atol=atol):
            logger.warning(f"Parameter data mismatch for {name1}")
            return False

        # Check gradients only if requested
        if check_gradients:
            if param1.grad is not None and param2.grad is not None:
                if not torch.allclose(param1.grad, param2.grad, atol=atol):
                    logger.warning(f"Gradient mismatch for {name1}")
                    return False

    # Check optimizer state
    if optimizer1 and optimizer2:
        state1 = optimizer1.state_dict()
        state2 = optimizer2.state_dict()

        if state1.keys() != state2.keys():
            logger.warning("Optimizer state key mismatch")
            return False

        for key in state1:
            if isinstance(state1[key], dict) and isinstance(state2[key], dict):
                for subkey in state1[key]:
                    val1, val2 = state1[key][subkey], state2[key][subkey]
                    if isinstance(val1, torch.Tensor):
                        if not torch.allclose(val1, val2, atol=atol):
                            logger.warning(f"Optimizer state mismatch: {key}.{subkey}")
                            return False

    return True


# =============================================================================
# Registry
# =============================================================================


CHECKPOINT_COMPONENTS = {
    'CheckpointPolicy': CheckpointPolicy,
    'CheckpointConfig': CheckpointConfig,
    'CheckpointState': CheckpointState,
    'CheckpointManager': CheckpointManager,
    'ResumeTrainer': ResumeTrainer,
    'create_checkpoint_manager': create_checkpoint_manager,
    'save_checkpoint': save_checkpoint,
    'load_checkpoint': load_checkpoint,
    'verify_gradient_consistency': verify_gradient_consistency,
}
