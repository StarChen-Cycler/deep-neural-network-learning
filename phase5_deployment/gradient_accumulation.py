"""
Gradient Accumulation Implementation for Memory-Efficient Training.

This module provides:
    - GradientAccumulator: Core gradient accumulation logic
    - GradientAccumulationTrainer: Integrated training loop with accumulation
    - Memory benchmark utilities
    - Dynamic accumulation step adjustment

Theory:
    Gradient Accumulation:
        - Accumulates gradients over multiple forward/backward passes
        - Only updates weights after accumulating N batches
        - Effective batch size = batch_size * accumulation_steps
        - Memory reduction: Only stores batch_size samples at a time

    Mathematical Equivalence:
        - Single batch of size B: gradient = sum(grad_i) / B
        - Accumulated N batches of size B/N:
          gradient = sum(sum(grad_ij) / (B/N)) / N = sum(grad_i) / B
        - Result: Identical gradients, lower peak memory

    Memory Savings:
        - Peak memory with batch_size=B: O(B * model_activations)
        - Peak memory with batch_size=B/N, accumulation=N: O(B/N * model_activations)
        - Savings: ~N times less peak activation memory

    For 4GB VRAM:
        - Recommended: batch_size=1-4, accumulation_steps=16-32
        - This achieves effective batch_size=16-128 with minimal memory

References:
    - PyTorch AMP Examples: https://pytorch.org/docs/stable/notes/amp_examples.html
    - Gradient Accumulation: https://kozodoi.me/blog/20210308/gradient-accumulation
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
import gc
import time
import logging

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    autocast = None
    GradScaler = None
    DataLoader = None
    Dataset = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GradientAccumulationConfig:
    """
    Configuration for gradient accumulation training.

    Attributes:
        accumulation_steps: Number of batches to accumulate before update
        batch_size: Micro-batch size (actual batch loaded at once)
        effective_batch_size: Target effective batch size (batch_size * accumulation_steps)
        use_amp: Whether to use automatic mixed precision
        max_grad_norm: Maximum gradient norm for clipping (0 to disable)
        gradient_accumulation_dtype: Data type for gradient accumulation
        dynamic_accumulation: Enable dynamic adjustment based on memory
        memory_threshold_mb: Memory threshold for dynamic adjustment
    """
    accumulation_steps: int = 4
    batch_size: int = 32
    effective_batch_size: Optional[int] = None
    use_amp: bool = True
    max_grad_norm: float = 1.0
    gradient_accumulation_dtype: str = "float32"
    dynamic_accumulation: bool = False
    memory_threshold_mb: float = 3500.0  # For 4GB GPU, leave ~500MB buffer

    def __post_init__(self):
        """Calculate effective batch size if not provided."""
        if self.effective_batch_size is None:
            self.effective_batch_size = self.batch_size * self.accumulation_steps
        else:
            # Calculate accumulation steps from effective batch size
            if self.effective_batch_size % self.batch_size != 0:
                raise ValueError(
                    f"effective_batch_size ({self.effective_batch_size}) must be "
                    f"divisible by batch_size ({self.batch_size})"
                )
            self.accumulation_steps = self.effective_batch_size // self.batch_size

    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        if self.accumulation_steps < 1:
            errors.append("accumulation_steps must be >= 1")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.max_grad_norm < 0:
            errors.append("max_grad_norm must be >= 0")
        return errors


# =============================================================================
# Gradient Accumulator
# =============================================================================


class GradientAccumulator:
    """
    Core gradient accumulation logic.

    This class handles the accumulation of gradients over multiple batches,
    providing the correct scaling and update timing.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            with accumulator.accumulate():
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulator.accumulation_steps
                loss.backward()

            if accumulator.should_update():
                optimizer.step()
                optimizer.zero_grad()

    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=4)
        >>> for i in range(100):
        ...     with accumulator.accumulate():
        ...         loss = model(x) / 4
        ...         loss.backward()
        ...     if accumulator.should_update():
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """

    def __init__(
        self,
        accumulation_steps: int = 4,
        scaler: Optional[Any] = None,
    ):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of batches to accumulate
            scaler: Optional GradScaler for AMP (automatically handles scaling)
        """
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")

        self.accumulation_steps = accumulation_steps
        self._scaler = scaler
        self._step_count = 0
        self._is_accumulating = False

    @property
    def scaler(self) -> Optional[Any]:
        """Get the gradient scaler."""
        return self._scaler

    @scaler.setter
    def scaler(self, value: Optional[Any]) -> None:
        """Set the gradient scaler."""
        self._scaler = value

    @property
    def step_count(self) -> int:
        """Get current step count within accumulation cycle."""
        return self._step_count

    @property
    def is_first_step(self) -> bool:
        """Check if this is the first step in accumulation cycle."""
        return self._step_count == 0

    @property
    def is_last_step(self) -> bool:
        """Check if this is the last step (update needed)."""
        return (self._step_count + 1) % self.accumulation_steps == 0

    def should_update(self) -> bool:
        """
        Check if optimizer should update (accumulation complete).

        Returns:
            True if accumulation cycle is complete and optimizer should step
        """
        return self.is_last_step

    def accumulate(self):
        """
        Context manager for gradient accumulation.

        Increments step counter on exit. Use with 'with' statement.

        Yields:
            self for context management
        """
        return self._AccumulationContext(self)

    def advance(self) -> bool:
        """
        Manually advance the accumulation counter.

        Returns:
            True if accumulation cycle is complete
        """
        self._step_count += 1
        return self.should_update()

    def reset(self) -> None:
        """Reset the accumulation counter."""
        self._step_count = 0

    def set_accumulation_steps(self, steps: int) -> None:
        """
        Dynamically adjust accumulation steps.

        Args:
            steps: New accumulation steps value
        """
        if steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.accumulation_steps = steps
        self._step_count = self._step_count % steps

    class _AccumulationContext:
        """Internal context manager for accumulation."""

        def __init__(self, accumulator: 'GradientAccumulator'):
            self.accumulator = accumulator

        def __enter__(self):
            self.accumulator._is_accumulating = True
            return self.accumulator

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.accumulator._is_accumulating = False
            self.accumulator._step_count += 1
            return False


# =============================================================================
# Gradient Accumulation Trainer
# =============================================================================


class GradientAccumulationTrainer:
    """
    Complete training loop with gradient accumulation support.

    This trainer handles:
        - Gradient accumulation with proper scaling
        - Mixed precision training (AMP)
        - Gradient clipping
        - Logging and metrics tracking

    Example:
        >>> config = GradientAccumulationConfig(
        ...     accumulation_steps=4,
        ...     batch_size=32,
        ...     use_amp=True,
        ... )
        >>> trainer = GradientAccumulationTrainer(model, config)
        >>> history = trainer.train(
        ...     train_loader=train_loader,
        ...     optimizer=optimizer,
        ...     criterion=nn.CrossEntropyLoss(),
        ...     epochs=10,
        ... )
    """

    def __init__(
        self,
        model: Any,
        config: Optional[GradientAccumulationConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            config: Gradient accumulation configuration
            device: Device to use (default: cuda if available, else cpu)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GradientAccumulationTrainer")

        self.model = model
        self.config = config or GradientAccumulationConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Initialize AMP components
        if self.config.use_amp and HAS_TORCH:
            try:
                # Use new API if available (PyTorch 2.0+)
                self.scaler = torch.amp.GradScaler('cuda')
            except (TypeError, AttributeError):
                # Fallback to old API
                self.scaler = GradScaler()
        else:
            self.scaler = None
        self.accumulator = GradientAccumulator(
            accumulation_steps=self.config.accumulation_steps,
            scaler=self.scaler,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _forward_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable,
    ) -> torch.Tensor:
        """
        Execute a single forward step.

        Args:
            batch: Tuple of (inputs, targets)
            criterion: Loss function

        Returns:
            Scaled loss for accumulation
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass with optional AMP
        if self.config.use_amp and autocast is not None:
            with autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        # This ensures gradients are correctly averaged
        scaled_loss = loss / self.accumulator.accumulation_steps

        return scaled_loss

    def _backward_step(self, loss: torch.Tensor) -> None:
        """
        Execute backward pass with optional gradient scaling.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Execute optimizer step with gradient clipping and optional unscaling.

        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            # Unscale gradients for clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(optimizer)

            # Clip gradients
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            # Optimizer step with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Clip gradients without scaler
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            optimizer.step()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Optional epoch number for logging

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_updates = 0

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            loss = self._forward_step(batch, criterion)

            # Backward pass
            self._backward_step(loss)

            # Accumulate loss for logging (unscaled)
            batch_size = batch[0].size(0)
            total_loss += loss.item() * self.accumulator.accumulation_steps * batch_size
            total_samples += batch_size

            # Advance accumulator and check if we should update
            if self.accumulator.advance():
                self._optimizer_step(optimizer)
                optimizer.zero_grad()
                num_updates += 1
                self.global_step += 1

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'updates': num_updates,
            'samples': total_samples,
        }

    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List]:
        """
        Full training loop with gradient accumulation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epochs: Number of epochs to train
            val_loader: Optional validation data loader
            callbacks: Optional list of callback functions

        Returns:
            Training history dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'updates_per_epoch': [],
        }

        for epoch in range(epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )
            history['train_loss'].append(train_metrics['loss'])
            history['updates_per_epoch'].append(train_metrics['updates'])

            # Validate if provided
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))

                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}"
                )

            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, self, history)

        return history

    def evaluate(
        self,
        val_loader: DataLoader,
        criterion: Callable,
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.config.use_amp and autocast is not None:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total += targets.size(0)

                # Calculate accuracy for classification
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()

        metrics = {
            'loss': total_loss / total if total > 0 else 0.0,
        }

        if total > 0 and correct > 0:
            metrics['accuracy'] = correct / total

        return metrics

    def set_accumulation_steps(self, steps: int) -> None:
        """
        Dynamically adjust accumulation steps.

        Args:
            steps: New accumulation steps value
        """
        self.accumulator.set_accumulation_steps(steps)
        self.config.accumulation_steps = steps
        logger.info(f"Adjusted accumulation steps to {steps}")


# =============================================================================
# Memory Utilities
# =============================================================================


def get_memory_usage(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Args:
        device: Device to check (default: cuda:0)

    Returns:
        Dictionary with memory statistics in MB
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}

    if device is None:
        device = 'cuda:0'

    return {
        'allocated': torch.cuda.memory_allocated(device) / 1024 / 1024,
        'reserved': torch.cuda.memory_reserved(device) / 1024 / 1024,
        'max_allocated': torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


def reset_memory_stats(device: Optional[str] = None) -> None:
    """
    Reset GPU memory statistics.

    Args:
        device: Device to reset (default: cuda:0)
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    if device is None:
        device = 'cuda:0'

    torch.cuda.reset_peak_memory_stats(device)


def benchmark_memory_usage(
    model: nn.Module,
    sample_input: torch.Tensor,
    batch_sizes: List[int],
    accumulation_steps: List[int],
    criterion: Optional[Callable] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Benchmark memory usage for different batch sizes and accumulation steps.

    Args:
        model: Model to benchmark
        sample_input: Sample input tensor (single sample)
        batch_sizes: List of batch sizes to test
        accumulation_steps: List of accumulation steps to test
        criterion: Optional loss function
        device: Device to use (default: cuda if available)

    Returns:
        Dictionary with benchmark results
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for memory benchmarking")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = model.to(device)
    results = []

    for batch_size in batch_sizes:
        for acc_steps in accumulation_steps:
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                reset_memory_stats()

            # Create batch
            batch_input = sample_input.unsqueeze(0).repeat(batch_size, *([1] * (sample_input.dim())))
            batch_target = torch.randint(0, 10, (batch_size,))

            # Reset model
            model.zero_grad()

            # Record initial memory
            initial_memory = get_memory_usage(device)

            try:
                # Forward pass
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)

                output = model(batch_input)

                if criterion is not None:
                    loss = criterion(output, batch_target) / acc_steps
                    loss.backward()

                # Record peak memory
                peak_memory = get_memory_usage(device)

                results.append({
                    'batch_size': batch_size,
                    'accumulation_steps': acc_steps,
                    'effective_batch_size': batch_size * acc_steps,
                    'initial_memory_mb': initial_memory['allocated'],
                    'peak_memory_mb': peak_memory['max_allocated'],
                    'memory_increase_mb': peak_memory['max_allocated'] - initial_memory['allocated'],
                    'success': True,
                })

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    results.append({
                        'batch_size': batch_size,
                        'accumulation_steps': acc_steps,
                        'effective_batch_size': batch_size * acc_steps,
                        'initial_memory_mb': initial_memory['allocated'],
                        'peak_memory_mb': None,
                        'memory_increase_mb': None,
                        'success': False,
                        'error': 'OOM',
                    })
                else:
                    raise

            finally:
                # Clean up
                del batch_input, batch_target
                if 'output' in locals():
                    del output
                if 'loss' in locals():
                    del loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return {
        'results': results,
        'device': str(device),
        'model_parameters': sum(p.numel() for p in model.parameters()),
    }


# =============================================================================
# Verification Functions
# =============================================================================


def verify_gradient_equivalence(
    model_class: type,
    input_shape: Tuple[int, ...],
    effective_batch_size: int = 128,
    accumulation_steps: int = 4,
    device: Optional[str] = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verify that gradient accumulation produces equivalent gradients.

    This function compares:
    1. Single batch of size effective_batch_size
    2. Accumulated batches of size effective_batch_size / accumulation_steps

    Args:
        model_class: Model class to instantiate
        input_shape: Shape of input tensor (without batch dimension)
        effective_batch_size: Target effective batch size
        accumulation_steps: Number of accumulation steps
        device: Device to use
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with verification results
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for gradient verification")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    micro_batch_size = effective_batch_size // accumulation_steps

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create two identical models
    model_single = model_class().to(device)
    model_accum = model_class().to(device)

    # Copy weights
    model_accum.load_state_dict(model_single.state_dict())

    # Create dummy data (same for both)
    torch.manual_seed(123)
    x_single = torch.randn(effective_batch_size, *input_shape, device=device)
    y_single = torch.randint(0, 10, (effective_batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()

    # === Single batch approach ===
    model_single.zero_grad()
    output_single = model_single(x_single)
    loss_single = criterion(output_single, y_single)
    loss_single.backward()

    # Store gradients
    grads_single = {}
    for name, param in model_single.named_parameters():
        if param.grad is not None:
            grads_single[name] = param.grad.clone()

    # === Accumulated approach ===
    model_accum.zero_grad()
    for i in range(accumulation_steps):
        start_idx = i * micro_batch_size
        end_idx = (i + 1) * micro_batch_size

        x_micro = x_single[start_idx:end_idx]
        y_micro = y_single[start_idx:end_idx]

        output_micro = model_accum(x_micro)
        loss_micro = criterion(output_micro, y_micro) / accumulation_steps
        loss_micro.backward()

    # Store gradients
    grads_accum = {}
    for name, param in model_accum.named_parameters():
        if param.grad is not None:
            grads_accum[name] = param.grad.clone()

    # Compare gradients
    comparison = {}
    all_close = True
    max_diff = 0.0

    for name in grads_single:
        if name in grads_accum:
            is_close = torch.allclose(
                grads_single[name],
                grads_accum[name],
                rtol=rtol,
                atol=atol,
            )
            diff = torch.abs(grads_single[name] - grads_accum[name]).max().item()

            comparison[name] = {
                'close': is_close,
                'max_diff': diff,
                'single_norm': grads_single[name].norm().item(),
                'accum_norm': grads_accum[name].norm().item(),
            }

            all_close = all_close and is_close
            max_diff = max(max_diff, diff)

    return {
        'equivalent': all_close,
        'max_difference': max_diff,
        'effective_batch_size': effective_batch_size,
        'accumulation_steps': accumulation_steps,
        'micro_batch_size': micro_batch_size,
        'comparison': comparison,
        'tolerance': {'rtol': rtol, 'atol': atol},
    }


def calculate_memory_savings(
    original_batch_size: int,
    accumulation_steps: int,
    model_activation_memory_per_sample: float,
) -> Dict[str, float]:
    """
    Calculate theoretical memory savings from gradient accumulation.

    Args:
        original_batch_size: Original batch size without accumulation
        accumulation_steps: Number of accumulation steps
        model_activation_memory_per_sample: Memory per sample for activations (MB)

    Returns:
        Dictionary with memory calculations
    """
    micro_batch_size = original_batch_size // accumulation_steps

    original_memory = original_batch_size * model_activation_memory_per_sample
    accumulated_memory = micro_batch_size * model_activation_memory_per_sample

    savings = original_memory - accumulated_memory
    savings_percent = (savings / original_memory) * 100 if original_memory > 0 else 0

    return {
        'original_batch_size': original_batch_size,
        'micro_batch_size': micro_batch_size,
        'accumulation_steps': accumulation_steps,
        'original_memory_mb': original_memory,
        'accumulated_memory_mb': accumulated_memory,
        'savings_mb': savings,
        'savings_percent': savings_percent,
        'effective_batch_size': original_batch_size,
    }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_gradient_accumulation_trainer(
    model: nn.Module,
    effective_batch_size: int = 128,
    micro_batch_size: int = 32,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
    device: Optional[str] = None,
) -> GradientAccumulationTrainer:
    """
    Convenience function to create a configured trainer.

    Args:
        model: Model to train
        effective_batch_size: Target effective batch size
        micro_batch_size: Actual batch size loaded at once
        use_amp: Whether to use mixed precision
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to use

    Returns:
        Configured GradientAccumulationTrainer
    """
    config = GradientAccumulationConfig(
        batch_size=micro_batch_size,
        effective_batch_size=effective_batch_size,
        use_amp=use_amp,
        max_grad_norm=max_grad_norm,
    )

    return GradientAccumulationTrainer(model, config, device)


def recommend_accumulation_settings(
    vram_gb: float = 4.0,
    model_params_m: float = 10.0,
    input_size_kb: float = 150.0,
) -> Dict[str, Any]:
    """
    Recommend gradient accumulation settings based on hardware.

    Args:
        vram_gb: Available GPU memory in GB
        model_params_m: Model parameters in millions
        input_size_kb: Input size per sample in KB

    Returns:
        Dictionary with recommended settings
    """
    # Estimate memory requirements
    # Model weights: ~4 bytes per parameter (FP32) or 2 bytes (FP16)
    model_memory_mb = model_params_m * 4  # FP32

    # Rough estimate: activations are typically 2-3x model size during training
    activation_overhead = 2.5
    training_memory_mb = model_memory_mb * activation_overhead

    # Available memory for batch
    buffer_mb = 500  # Leave buffer for CUDA overhead
    available_for_batch = (vram_gb * 1024) - training_memory_mb - buffer_mb

    # Estimate samples that fit
    sample_memory_mb = input_size_kb / 1024
    max_batch_size = max(1, int(available_for_batch / (sample_memory_mb * activation_overhead)))

    # Recommend settings for common effective batch sizes
    recommendations = []
    for effective_batch in [32, 64, 128, 256]:
        if effective_batch % max_batch_size == 0:
            acc_steps = effective_batch // max_batch_size
        else:
            # Find closest divisor
            acc_steps = 1
            for s in range(1, effective_batch + 1):
                if effective_batch % s == 0 and s <= max_batch_size:
                    acc_steps = effective_batch // s
                    break

        micro_batch = effective_batch // acc_steps
        if micro_batch <= max_batch_size:
            recommendations.append({
                'effective_batch_size': effective_batch,
                'micro_batch_size': micro_batch,
                'accumulation_steps': acc_steps,
            })

    return {
        'vram_gb': vram_gb,
        'estimated_model_memory_mb': model_memory_mb,
        'estimated_training_memory_mb': training_memory_mb,
        'recommended_max_batch_size': max_batch_size,
        'recommendations': recommendations,
        'notes': [
            f"For {vram_gb}GB VRAM, use batch_size=1-{max_batch_size}",
            "Enable AMP (use_amp=True) to reduce memory further",
            "Use gradient checkpointing for very large models",
        ],
    }


# =============================================================================
# Module Exports
# =============================================================================


GRADIENT_ACCUMULATION_COMPONENTS = {
    'GradientAccumulationConfig': GradientAccumulationConfig,
    'GradientAccumulator': GradientAccumulator,
    'GradientAccumulationTrainer': GradientAccumulationTrainer,
    'GradientAccumulationConfig': GradientAccumulationConfig,
}

UTIL_FUNCTIONS = {
    'get_memory_usage': get_memory_usage,
    'reset_memory_stats': reset_memory_stats,
    'benchmark_memory_usage': benchmark_memory_usage,
    'verify_gradient_equivalence': verify_gradient_equivalence,
    'calculate_memory_savings': calculate_memory_savings,
    'create_gradient_accumulation_trainer': create_gradient_accumulation_trainer,
    'recommend_accumulation_settings': recommend_accumulation_settings,
}

__all__ = list(GRADIENT_ACCUMULATION_COMPONENTS.keys()) + list(UTIL_FUNCTIONS.keys())
