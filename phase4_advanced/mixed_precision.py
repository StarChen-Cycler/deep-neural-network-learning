"""
Mixed Precision Training: FP16/BF16/TF32 Implementation.

This module provides:
    - MixedPrecisionTrainer for automatic mixed precision training
    - GradScaler wrapper with dynamic loss scaling
    - Precision mode detection (FP16/BF16/TF32)
    - Memory-efficient training utilities

Theory:
    Mixed Precision Training:
        - Uses FP16 for compute-intensive operations (conv, matmul)
        - Keeps FP32 for numerically sensitive ops (BatchNorm, Softmax, Loss)
        - Reduces memory by ~50%, speeds up training 1.5-3x on Tensor Core GPUs

    Loss Scaling:
        - FP16 has limited range (max ~65504)
        - Small gradients can underflow to zero
        - Solution: Scale loss up before backward, scale gradients down after
        - Dynamic scaling: Adjust scale based on gradient health

    Precision Types:
        - FP16 (half): 16-bit float, range ±65504, precision ~3 decimal digits
        - BF16 (bfloat16): 16-bit brain float, same range as FP32, lower precision
        - TF32 (tensorfloat32): 19-bit format on Ampere+ GPUs, ~10x faster than FP32

References:
    - Mixed Precision Training (Micikevicius et al., 2017)
    - Automatic Mixed Precision package: https://pytorch.org/docs/stable/amp.html
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import logging

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    # Use new API (PyTorch 2.0+)
    try:
        from torch.amp import autocast, GradScaler
        _AMP_DEVICE_TYPE = 'cuda'
    except ImportError:
        # Fallback to deprecated API (PyTorch < 2.0)
        from torch.cuda.amp import autocast, GradScaler
        _AMP_DEVICE_TYPE = None
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    autocast = None
    GradScaler = None
    _AMP_DEVICE_TYPE = None

logger = logging.getLogger(__name__)


# =============================================================================
# Precision Detection Utilities
# =============================================================================


def is_fp16_supported() -> bool:
    """
    Check if FP16 is supported on current device.

    Returns:
        True if CUDA is available and supports FP16
    """
    if not HAS_TORCH:
        return False
    if not torch.cuda.is_available():
        return False
    # All CUDA devices support FP16
    return True


def is_bf16_supported() -> bool:
    """
    Check if BF16 (bfloat16) is supported on current device.

    BF16 requires:
        - CUDA availability
        - Ampere architecture (SM >= 8.0) or newer

    Returns:
        True if BF16 is supported
    """
    if not HAS_TORCH:
        return False
    if not torch.cuda.is_available():
        return False

    # Check compute capability >= 8.0 (Ampere)
    try:
        major, minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def is_tf32_supported() -> bool:
    """
    Check if TF32 (tensorfloat32) is supported on current device.

    TF32 is available on Ampere+ GPUs (SM >= 8.0).

    Returns:
        True if TF32 is supported
    """
    return is_bf16_supported()  # Same hardware requirement


def enable_tf32(enabled: bool = True) -> None:
    """
    Enable or disable TF32 mode on Ampere+ GPUs.

    TF32 provides ~10x speedup for matmul operations while maintaining
    reasonable numerical accuracy.

    Args:
        enabled: Whether to enable TF32

    Note:
        TF32 is enabled by default in PyTorch 1.7+ on Ampere+ GPUs.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available, cannot enable TF32")
        return

    if not torch.cuda.is_available():
        return

    # Enable/disable TF32 for matmul and cudnn
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled

    if enabled:
        logger.info("TF32 enabled for CUDA matmul and cuDNN")
    else:
        logger.info("TF32 disabled")


def get_recommended_precision() -> str:
    """
    Get recommended precision mode based on hardware.

    Returns:
        'bf16' if supported, else 'fp16', else 'fp32'
    """
    if is_bf16_supported():
        return 'bf16'
    elif is_fp16_supported():
        return 'fp16'
    else:
        return 'fp32'


def get_device_info() -> Dict[str, Any]:
    """
    Get device information for mixed precision support.

    Returns:
        Dictionary with device info
    """
    info = {
        'cuda_available': False,
        'device_name': 'CPU',
        'compute_capability': None,
        'fp16_supported': False,
        'bf16_supported': False,
        'tf32_supported': False,
        'tensor_cores': False,
        'vram_gb': 0.0,
    }

    if not HAS_TORCH:
        info['error'] = 'PyTorch not installed'
        return info

    info['cuda_available'] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        try:
            major, minor = torch.cuda.get_device_capability()
            info['compute_capability'] = f'{major}.{minor}'
            info['tensor_cores'] = major >= 7  # Volta+
        except Exception:
            pass

        info['fp16_supported'] = is_fp16_supported()
        info['bf16_supported'] = is_bf16_supported()
        info['tf32_supported'] = is_tf32_supported()

    return info


# =============================================================================
# GradScaler Wrapper
# =============================================================================


@dataclass
class GradScalerConfig:
    """
    Configuration for GradScaler.

    Attributes:
        init_scale: Initial scale factor (default: 2^16 = 65536)
        growth_factor: Factor to multiply scale by when increasing
        backoff_factor: Factor to multiply scale by when decreasing
        growth_interval: Number of steps between scale increases
        enabled: Whether gradient scaling is enabled
    """
    init_scale: float = 2.0 ** 16  # 65536
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True


class MixedPrecisionScaler:
    """
    Wrapper around torch.cuda.amp.GradScaler with additional features.

    Provides:
        - Dynamic loss scaling
        - Gradient health monitoring
        - NaN/Inf detection
        - Scale adjustment logging
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        """
        Initialize the scaler.

        Args:
            init_scale: Initial scale factor
            growth_factor: Factor to increase scale by
            backoff_factor: Factor to decrease scale by (on inf/nan)
            growth_interval: Steps between scale increases
            enabled: Whether to enable gradient scaling
        """
        self.enabled = enabled and HAS_TORCH

        if self.enabled:
            # Use new API if available (PyTorch 2.0+)
            if _AMP_DEVICE_TYPE == 'cuda':
                self._scaler = GradScaler(
                    device='cuda',
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                    enabled=True,
                )
            else:
                # Fallback to deprecated API
                self._scaler = GradScaler(
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                    enabled=True,
                )
        else:
            self._scaler = None

        self._init_scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

        # Statistics
        self._step_count = 0
        self._scale_history: List[float] = []
        self._found_inf_count = 0

    @property
    def scale(self) -> float:
        """Get current scale factor."""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0

    @property
    def found_inf(self) -> bool:
        """Check if inf/nan was found in the last step."""
        return self._found_inf_count > 0

    def scale_loss(self, loss: 'torch.Tensor') -> 'torch.Tensor':
        """
        Scale the loss for backward pass.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss tensor
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: 'torch.optim.Optimizer') -> Optional[float]:
        """
        Step the optimizer with unscaled gradients.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Loss value if step was taken, None if skipped
        """
        if self._scaler is not None:
            return self._scaler.step(optimizer)
        else:
            optimizer.step()
            return None

    def update(self) -> None:
        """
        Update the scaler for the next iteration.

        Should be called after optimizer.step().
        """
        if self._scaler is not None:
            old_scale = self._scaler.get_scale()
            self._scaler.update()
            new_scale = self._scaler.get_scale()

            # Track scale changes
            self._scale_history.append(new_scale)

            # Detect if scale was reduced (found inf/nan)
            if new_scale < old_scale:
                self._found_inf_count += 1
                logger.warning(
                    f"Gradient inf/nan detected, scale reduced: "
                    f"{old_scale:.0f} -> {new_scale:.0f}"
                )

        self._step_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scaler statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'current_scale': self.scale,
            'init_scale': self._init_scale,
            'step_count': self._step_count,
            'found_inf_count': self._found_inf_count,
            'scale_history': self._scale_history[-100:],  # Last 100 values
            'enabled': self.enabled,
        }

    def is_health_check_passed(self, min_scale: float = 1.0) -> bool:
        """
        Check if gradient scaling is healthy.

        Args:
            min_scale: Minimum acceptable scale

        Returns:
            True if healthy
        """
        return self.scale >= min_scale


# =============================================================================
# Mixed Precision Trainer
# =============================================================================


class MixedPrecisionTrainer:
    """
    Trainer with automatic mixed precision support.

    Features:
        - Automatic autocast for forward pass
        - Gradient scaling for backward pass
        - FP16/BF16/FP32 mode selection
        - Memory usage tracking
        - Gradient health monitoring

    Usage:
        trainer = MixedPrecisionTrainer(model, optimizer, criterion)

        for epoch in range(epochs):
            for batch in dataloader:
                loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: 'nn.Module',
        optimizer: 'torch.optim.Optimizer',
        criterion: Callable,
        precision: str = 'auto',
        scaler_config: Optional[GradScalerConfig] = None,
        grad_clip_norm: Optional[float] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            precision: 'fp16', 'bf16', 'fp32', or 'auto'
            scaler_config: Configuration for gradient scaler
            grad_clip_norm: Optional gradient clipping norm
            device: Device to use (default: cuda if available, else cpu)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for MixedPrecisionTrainer")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Determine precision mode
        if precision == 'auto':
            self.precision = get_recommended_precision()
        else:
            self.precision = precision

        # Validate precision
        if self.precision == 'bf16' and not is_bf16_supported():
            logger.warning("BF16 not supported, falling back to FP16")
            self.precision = 'fp16'

        if self.precision == 'fp16' and not is_fp16_supported():
            logger.warning("FP16 not supported, falling back to FP32")
            self.precision = 'fp32'

        # Initialize scaler for FP16/BF16
        self.use_amp = self.precision in ('fp16', 'bf16')

        if scaler_config is None:
            scaler_config = GradScalerConfig()

        self.scaler = MixedPrecisionScaler(
            init_scale=scaler_config.init_scale,
            growth_factor=scaler_config.growth_factor,
            backoff_factor=scaler_config.backoff_factor,
            growth_interval=scaler_config.growth_interval,
            enabled=self.use_amp,
        )

        # Get dtype for autocast
        if self.precision == 'fp16':
            self.amp_dtype = torch.float16
        elif self.precision == 'bf16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32

        # Training statistics
        self._step_count = 0
        self._epoch_count = 0
        self._loss_history: List[float] = []
        self._memory_history: List[float] = []

        logger.info(
            f"MixedPrecisionTrainer initialized: "
            f"precision={self.precision}, device={self.device}, "
            f"amp_dtype={self.amp_dtype}"
        )

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    def train_step(
        self,
        inputs: 'torch.Tensor',
        targets: 'torch.Tensor',
    ) -> float:
        """
        Perform a single training step with mixed precision.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass with autocast
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass with scaled gradients
            self.scaler.scale_loss(loss).backward()

            # Gradient clipping
            if self.grad_clip_norm is not None:
                self.scaler._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard FP32 training
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.optimizer.step()

        # Record statistics
        loss_value = loss.item()
        self._loss_history.append(loss_value)
        self._memory_history.append(self._get_memory_mb())
        self._step_count += 1

        return loss_value

    def eval_step(
        self,
        inputs: 'torch.Tensor',
        targets: 'torch.Tensor',
    ) -> Tuple[float, 'torch.Tensor']:
        """
        Perform a single evaluation step.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Tuple of (loss, outputs)
        """
        self.model.eval()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

        return loss.item(), outputs

    def train_epoch(
        self,
        dataloader: 'torch.utils.data.DataLoader',
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            loss = self.train_step(inputs, targets)
            epoch_losses.append(loss)

        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        self._epoch_count += 1

        return {
            'epoch': self._epoch_count,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'memory_mb': self._memory_history[-1] if self._memory_history else 0,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.

        Returns:
            Dictionary with training statistics
        """
        stats = {
            'step_count': self._step_count,
            'epoch_count': self._epoch_count,
            'precision': self.precision,
            'device': str(self.device),
            'scaler_stats': self.scaler.get_stats() if self.scaler else None,
            'loss_history': self._loss_history[-100:],
            'memory_history': self._memory_history[-100:],
        }

        if self._loss_history:
            stats['loss_mean'] = sum(self._loss_history) / len(self._loss_history)
            stats['loss_min'] = min(self._loss_history)
            stats['loss_max'] = max(self._loss_history)

        if self._memory_history:
            stats['memory_max_mb'] = max(self._memory_history)

        return stats


# =============================================================================
# Utility Functions
# =============================================================================


def compare_precision_modes(
    model: 'nn.Module',
    dataloader: 'torch.utils.data.DataLoader',
    criterion: Callable,
    optimizer_class: type,
    num_steps: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare training with different precision modes.

    Args:
        model: Model to train (will be copied for each mode)
        dataloader: Training data
        criterion: Loss function
        optimizer_class: Optimizer class
        num_steps: Number of steps to run

    Returns:
        Dictionary with results for each precision mode
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    results = {}
    modes = ['fp32']

    if is_fp16_supported():
        modes.append('fp16')
    if is_bf16_supported():
        modes.append('bf16')

    for mode in modes:
        logger.info(f"Testing precision mode: {mode}")

        # Reset model
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())

        optimizer = optimizer_class(model_copy.parameters(), lr=0.001)

        trainer = MixedPrecisionTrainer(
            model=model_copy,
            optimizer=optimizer,
            criterion=criterion,
            precision=mode,
        )

        losses = []
        times = []
        memory = []

        data_iter = iter(dataloader)
        for step in range(num_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            start_time = time.time()
            loss = trainer.train_step(inputs, targets)
            step_time = time.time() - start_time

            losses.append(loss)
            times.append(step_time)
            memory.append(trainer._get_memory_mb())

        results[mode] = {
            'final_loss': losses[-1],
            'loss_improvement': losses[0] - losses[-1],
            'avg_step_time': sum(times) / len(times),
            'total_time': sum(times),
            'max_memory_mb': max(memory),
            'loss_history': losses,
        }

        logger.info(
            f"  {mode}: loss={losses[-1]:.4f}, "
            f"time={results[mode]['avg_step_time']*1000:.2f}ms, "
            f"memory={results[mode]['max_memory_mb']:.1f}MB"
        )

    return results


def enable_optimizations_for_small_vram() -> Dict[str, bool]:
    """
    Enable optimizations for small VRAM (4GB).

    Returns:
        Dictionary of enabled optimizations
    """
    optimizations = {}

    if not HAS_TORCH or not torch.cuda.is_available():
        return optimizations

    # Enable TF32 (if supported) for faster compute
    if is_tf32_supported():
        enable_tf32(True)
        optimizations['tf32'] = True

    # Disable FP16 reduction accumulation (more stable)
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        optimizations['fp16_reduced_precision_disabled'] = True
    except Exception:
        pass

    # Set cudnn to deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    optimizations['cudnn_deterministic'] = True

    # Empty cache
    torch.cuda.empty_cache()
    optimizations['cache_cleared'] = True

    logger.info(f"Enabled optimizations for small VRAM: {optimizations}")
    return optimizations


# =============================================================================
# Registry
# =============================================================================

MIXED_PRECISION_MODES = {
    'fp16': {
        'dtype': 'float16',
        'bytes_per_element': 2,
        'description': 'Half precision, 50% memory savings',
    },
    'bf16': {
        'dtype': 'bfloat16',
        'bytes_per_element': 2,
        'description': 'Brain float, same range as FP32, lower precision',
    },
    'fp32': {
        'dtype': 'float32',
        'bytes_per_element': 4,
        'description': 'Single precision, maximum accuracy',
    },
    'tf32': {
        'dtype': 'tensorfloat32',
        'bytes_per_element': 4,
        'description': 'Tensor float, ~10x faster than FP32 on Ampere+',
    },
}


def get_precision_info(mode: str) -> Dict[str, Any]:
    """
    Get information about a precision mode.

    Args:
        mode: Precision mode name

    Returns:
        Dictionary with mode information
    """
    if mode not in MIXED_PRECISION_MODES:
        available = list(MIXED_PRECISION_MODES.keys())
        raise ValueError(f"Unknown mode: {mode}. Available: {available}")
    return MIXED_PRECISION_MODES[mode]
