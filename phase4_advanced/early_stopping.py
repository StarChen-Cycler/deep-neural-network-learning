"""
Early Stopping Callback for PyTorch Training.

This module provides:
    - EarlyStopping callback class
    - Patience-based training termination
    - Best model checkpoint saving
    - Restore best weights functionality

Theory:
    Early Stopping:
        - Monitor validation loss during training
        - Stop training when loss stops improving for N epochs (patience)
        - Prevents overfitting and saves training time
        - Restores model to best weights after stopping

    Patience Mechanism:
        - Counter increases when loss doesn't improve
        - Counter resets when new best loss is found
        - Training stops when counter >= patience

    Best Model Saving:
        - Save checkpoint when validation loss improves
        - Can restore best weights at end of training
        - Avoids keeping overfitted final model

References:
    - Early Stopping: https://deeplearning.stanford.edu/tutorial/
"""

from typing import Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
import os
import time
import logging
import warnings

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# =============================================================================
# Early Stopping Callback
# =============================================================================


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for EarlyStopping callback.

    Attributes:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
        restore_best_weights: Whether to restore best model weights on stop
        verbose: Whether to print messages
        checkpoint_path: Path to save best model checkpoint (optional)
        save_checkpoint: Whether to save checkpoint on improvement
    """
    patience: int = 5
    min_delta: float = 0.0
    mode: str = 'min'
    restore_best_weights: bool = True
    verbose: bool = True
    checkpoint_path: Optional[str] = None
    save_checkpoint: bool = False


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Features:
        - Patience-based stopping
        - Minimum delta threshold for improvement
        - Best model checkpoint saving
        - Restore best weights on stop
        - Mode support (min for loss, max for accuracy)

    Usage:
        early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

        for epoch in range(epochs):
            train_loss = train_one_epoch()
            val_loss = validate()

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Optionally restore best weights
        if early_stopping.restore_best_weights:
            model.load_state_dict(early_stopping.best_weights)
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
        save_checkpoint: bool = False,
        config: Optional[EarlyStoppingConfig] = None,
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for metrics like loss, 'max' for metrics like accuracy
            restore_best_weights: Whether to restore best weights on stop
            verbose: Whether to print messages
            checkpoint_path: Path to save checkpoint
            save_checkpoint: Whether to save checkpoint on improvement
            config: Configuration object (overrides individual args)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for EarlyStopping")

        # Use config if provided
        if config is not None:
            patience = config.patience
            min_delta = config.min_delta
            mode = config.mode
            restore_best_weights = config.restore_best_weights
            verbose = config.verbose
            checkpoint_path = config.checkpoint_path
            save_checkpoint = config.save_checkpoint

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.save_checkpoint = save_checkpoint

        # Validate mode
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        # Initialize state
        self._reset_state()

        # Mode-specific comparison
        if mode == 'min':
            self._is_improvement = lambda current, best: current < best - min_delta
            self._best_value = float('inf')
        else:
            self._is_improvement = lambda current, best: current > best + min_delta
            self._best_value = float('-inf')

    def _reset_state(self) -> None:
        """Reset internal state."""
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights: Optional[Dict[str, Any]] = None
        self._best_value = float('inf') if self.mode == 'min' else float('-inf')
        self._epoch_count = 0

    @property
    def best_score(self) -> float:
        """Get best score (for backward compatibility)."""
        return self._best_value

    def __call__(
        self,
        metric: float,
        model: Optional['nn.Module'] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Call early stopping with current metric.

        Args:
            metric: Current metric value (e.g., validation loss)
            model: Model to save weights from (required if restore_best_weights=True)
            epoch: Current epoch number (optional, for logging)

        Returns:
            True if early stopping should be triggered
        """
        return self.step(metric, model, epoch)

    def step(
        self,
        metric: float,
        model: Optional['nn.Module'] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Check if early stopping should be triggered.

        Args:
            metric: Current metric value (e.g., validation loss)
            model: Model to save weights from (required if restore_best_weights=True)
            epoch: Current epoch number (optional, for logging)

        Returns:
            True if early stopping should be triggered
        """
        self._epoch_count += 1
        current_epoch = epoch if epoch is not None else self._epoch_count

        if self._is_improvement(metric, self._best_value):
            # Improvement found
            self._best_value = metric
            self.best_epoch = current_epoch
            self.counter = 0

            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            # Save checkpoint
            if self.save_checkpoint and model is not None:
                self._save_checkpoint(model, current_epoch, metric)

            if self.verbose:
                logger.info(
                    f"Epoch {current_epoch}: {self.mode} improved to {metric:.6f}"
                )

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                logger.info(
                    f"Epoch {current_epoch}: {self.mode} did not improve. "
                    f"Patience: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered at epoch {current_epoch}. "
                        f"Best {self.mode}: {self._best_value:.6f} at epoch {self.best_epoch}"
                    )

        return self.early_stop

    def _save_checkpoint(
        self,
        model: 'nn.Module',
        epoch: int,
        metric: float,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch
            metric: Current metric value
        """
        if self.checkpoint_path is None:
            warnings.warn("checkpoint_path not set, cannot save checkpoint")
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            f'best_{self.mode}': metric,
            'early_stopping_state': {
                'counter': self.counter,
                'best_epoch': self.best_epoch,
                f'best_{self.mode}': self._best_value,
            }
        }

        os.makedirs(
            os.path.dirname(self.checkpoint_path)
            if os.path.dirname(self.checkpoint_path) else '.',
            exist_ok=True
        )
        torch.save(checkpoint, self.checkpoint_path)

        if self.verbose:
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")

    def restore_weights(self, model: 'nn.Module') -> None:
        """
        Restore model to best weights.

        Args:
            model: Model to restore weights to
        """
        if self.best_weights is None:
            warnings.warn("No best weights to restore")
            return

        model.load_state_dict(self.best_weights)

        if self.verbose:
            logger.info(
                f"Restored model to best weights from epoch {self.best_epoch}"
            )

    def reset(self) -> None:
        """Reset early stopping state for new training run."""
        self._reset_state()

        # Re-initialize comparison function
        if self.mode == 'min':
            self._best_value = float('inf')
        else:
            self._best_value = float('-inf')

        if self.verbose:
            logger.info("Early stopping state reset")

    def get_state(self) -> Dict[str, Any]:
        """
        Get early stopping state for serialization.

        Returns:
            Dictionary with state
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
            'best_value': self._best_value,
            'epoch_count': self._epoch_count,
            'restore_best_weights': self.restore_best_weights,
            'verbose': self.verbose,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load early stopping state.

        Args:
            state: Dictionary with state
        """
        self.patience = state.get('patience', self.patience)
        self.min_delta = state.get('min_delta', self.min_delta)
        self.mode = state.get('mode', self.mode)
        self.counter = state.get('counter', 0)
        self.early_stop = state.get('early_stop', False)
        self.best_epoch = state.get('best_epoch', 0)
        self._best_value = state.get('best_value', float('inf'))
        self._epoch_count = state.get('epoch_count', 0)

        # Re-initialize comparison function based on mode
        if self.mode == 'min':
            self._is_improvement = lambda current, best: current < best - self.min_delta
        else:
            self._is_improvement = lambda current, best: current > best + self.min_delta

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"mode='{self.mode}', counter={self.counter}, "
            f"best_epoch={self.best_epoch}, early_stop={self.early_stop})"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def create_early_stopping(
    patience: int = 5,
    min_delta: float = 0.0,
    mode: str = 'min',
    restore_best_weights: bool = True,
    verbose: bool = True,
) -> EarlyStopping:
    """
    Factory function to create EarlyStopping callback.

    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
        restore_best_weights: Whether to restore best weights on stop
        verbose: Whether to print messages

    Returns:
        EarlyStopping instance
    """
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )


def validate_early_stopping(
    model: 'nn.Module',
    val_loader,
    criterion,
    device: str = 'cuda',
) -> float:
    """
    Validate model and return loss for early stopping.

    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to use

    Returns:
        Average validation loss
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# Integration with Training Loop
# =============================================================================


class EarlyStoppingTrainer:
    """
    Trainer with built-in early stopping support.

    This class wraps a training loop with early stopping functionality.

    Usage:
        trainer = EarlyStoppingTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            patience=5,
        )

        history = trainer.train(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: 'nn.Module',
        optimizer: 'torch.optim.Optimizer',
        criterion: Callable,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            patience: Early stopping patience
            min_delta: Minimum delta for improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
            device: Device to use
            checkpoint_path: Path for checkpoints
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            restore_best_weights=restore_best_weights,
            checkpoint_path=checkpoint_path,
            save_checkpoint=checkpoint_path is not None,
        )

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training dataloader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_loader) -> float:
        """
        Run validation.

        Args:
            val_loader: Validation dataloader

        Returns:
            Average validation loss
        """
        return validate_early_stopping(
            self.model, val_loader, self.criterion, self.device
        )

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train with early stopping.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Maximum number of epochs
            verbose: Whether to print progress

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'stopped_early': False,
            'best_epoch': 0,
            'best_val_loss': float('inf'),
        }

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping check
            if self.early_stopping(val_loss, self.model, epoch + 1):
                history['stopped_early'] = True
                break

        # Restore best weights
        if self.early_stopping.restore_best_weights:
            self.early_stopping.restore_weights(self.model)

        history['best_epoch'] = self.early_stopping.best_epoch
        history['best_val_loss'] = self.early_stopping._best_value

        return history


# =============================================================================
# Registry
# =============================================================================

EARLY_STOPPING_COMPONENTS = {
    'callback': ['EarlyStopping', 'EarlyStoppingConfig'],
    'factory': ['create_early_stopping'],
    'utils': ['validate_early_stopping'],
    'trainer': ['EarlyStoppingTrainer'],
}
