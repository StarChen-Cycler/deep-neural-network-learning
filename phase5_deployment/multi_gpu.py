"""
Multi-GPU Configuration Utilities.

This module provides:
    - GPU detection and configuration
    - DataParallel (DP) wrapper utilities
    - Multi-GPU memory management
    - Device placement helpers
    - PyTorch Lightning strategy wrappers

Theory:
    Multi-GPU Strategies:
        - DataParallel (DP): Single process, multi-threaded
          - Simple to use: nn.DataParallel(model)
          - GIL bottleneck limits scaling
          - Good for small-scale testing

        - DistributedDataParallel (DDP): Multi-process
          - True parallelism, better scaling
          - Requires process spawning
          - Recommended for production

    Memory Management:
        - Each GPU stores full model + optimizer + gradients
        - Batch size per GPU = total_batch_size / num_gpus
        - Use gradient accumulation for large effective batches

References:
    - Multi-GPU Tutorial: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    - CUDA Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
"""

from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
import os
import logging

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DataParallel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    DataParallel = None

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Detection
# =============================================================================


def get_gpu_count() -> int:
    """
    Get number of available GPUs.

    Returns:
        Number of CUDA-capable GPUs
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get detailed information about all GPUs.

    Returns:
        List of dictionaries with GPU information
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'index': i,
            'name': props.name,
            'vram_gb': props.total_memory / (1024**3),
            'compute_capability': f'{props.major}.{props.minor}',
            'multi_processor_count': props.multi_processor_count,
        })

    return gpus


def get_recommended_batch_size(
    model: 'nn.Module',
    input_shape: tuple,
    dtype: str = 'float32',
    safety_factor: float = 0.8,
) -> int:
    """
    Estimate recommended batch size for a single GPU.

    Args:
        model: PyTorch model
        input_shape: Shape of a single input (without batch dimension)
        dtype: Data type ('float32', 'float16', 'bfloat16')
        safety_factor: Factor to multiply estimated size (0-1)

    Returns:
        Estimated batch size
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return 1

    # Get GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()

    # Estimate model memory (parameters + gradients + optimizer states)
    param_count = sum(p.numel() for p in model.parameters())

    dtype_bytes = {'float32': 4, 'float16': 2, 'bfloat16': 2}.get(dtype, 4)

    # Model: params + gradients + optimizer (Adam: 2x, SGD: 1x)
    model_memory = param_count * dtype_bytes * 4  # Conservative estimate

    # Activation memory per sample (rough estimate)
    sample_memory = int(np.prod(input_shape)) * dtype_bytes * 10  # 10x for activations

    # Estimate batch size
    available_for_batch = (free_memory - model_memory) * safety_factor
    batch_size = max(1, int(available_for_batch / sample_memory))

    return batch_size


# =============================================================================
# DataParallel Utilities
# =============================================================================


def wrap_data_parallel(
    model: 'nn.Module',
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    dim: int = 0,
) -> 'nn.DataParallel':
    """
    Wrap model with DataParallel for multi-GPU training.

    DataParallel is simpler than DDP but has GIL bottleneck.
    Use for small-scale multi-GPU testing.

    Args:
        model: PyTorch model
        device_ids: GPU indices to use (default: all available)
        output_device: GPU to collect outputs (default: device_ids[0])
        dim: Dimension along which to scatter

    Returns:
        DataParallel-wrapped model
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) == 0:
        raise ValueError("No GPUs available for DataParallel")

    if output_device is None:
        output_device = device_ids[0]

    model = model.to(f'cuda:{device_ids[0]}')

    return DataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        dim=dim,
    )


def unwrap_data_parallel(model: 'nn.Module') -> 'nn.Module':
    """
    Unwrap DataParallel to get underlying model.

    Args:
        model: Potentially DataParallel-wrapped model

    Returns:
        Underlying model
    """
    if isinstance(model, DataParallel):
        return model.module
    return model


# =============================================================================
# Device Placement
# =============================================================================


def get_device(device: Optional[Union[str, 'torch.device', int]] = None) -> 'torch.device':
    """
    Get torch device from various input formats.

    Args:
        device: Device specification (None, 'cuda', 'cuda:0', 0, torch.device)

    Returns:
        torch.device object
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(device, torch.device):
        return device

    if isinstance(device, int):
        return torch.device(f'cuda:{device}')

    return torch.device(device)


def to_device(
    data: Any,
    device: Union[str, 'torch.device', int],
    non_blocking: bool = True,
) -> Any:
    """
    Move data to device recursively.

    Handles tensors, lists, dicts, and tuples.

    Args:
        data: Data to move
        device: Target device
        non_blocking: Use non-blocking transfer

    Returns:
        Data on target device
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    device = get_device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [to_device(item, device, non_blocking) for item in data]
        return type(data)(moved)
    else:
        return data


# =============================================================================
# Memory Management
# =============================================================================


def get_memory_usage(device: int = 0) -> Dict[str, float]:
    """
    Get GPU memory usage statistics.

    Args:
        device: GPU index

    Returns:
        Dictionary with memory statistics in MB
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}

    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    free = total - reserved

    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total,
    }


def clear_cuda_cache() -> None:
    """Clear CUDA memory cache."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


def set_cuda_device(device: int) -> None:
    """
    Set current CUDA device.

    Args:
        device: GPU index
    """
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.set_device(device)


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class MultiGPUConfig:
    """
    Configuration for multi-GPU training.

    Attributes:
        strategy: 'dp' (DataParallel) or 'ddp' (DistributedDataParallel)
        device_ids: List of GPU indices (None = all available)
        batch_size_per_gpu: Batch size per GPU
        sync_batchnorm: Whether to use SyncBatchNorm
        pin_memory: Pin memory for faster data transfer
        find_unused_parameters: DDP find unused parameters
    """
    strategy: str = 'ddp'
    device_ids: Optional[List[int]] = None
    batch_size_per_gpu: int = 32
    sync_batchnorm: bool = True
    pin_memory: bool = True
    find_unused_parameters: bool = False

    def __post_init__(self):
        """Validate and set defaults."""
        if self.strategy not in ('dp', 'ddp'):
            raise ValueError(f"Invalid strategy: {self.strategy}")

        if self.device_ids is None and HAS_TORCH:
            self.device_ids = list(range(get_gpu_count()))

    @property
    def effective_batch_size(self) -> int:
        """Get total batch size across all GPUs."""
        num_gpus = len(self.device_ids) if self.device_ids else 1
        return self.batch_size_per_gpu * num_gpus

    @property
    def world_size(self) -> int:
        """Get world size for DDP."""
        return len(self.device_ids) if self.device_ids else 1


# =============================================================================
# Lightning Strategy Wrappers
# =============================================================================


def get_lightning_strategy(config: MultiGPUConfig) -> Dict[str, Any]:
    """
    Get PyTorch Lightning strategy configuration.

    Args:
        config: Multi-GPU configuration

    Returns:
        Dictionary with Lightning Trainer kwargs
    """
    if config.strategy == 'ddp':
        return {
            'strategy': 'ddp',
            'devices': config.device_ids,
            'accelerator': 'gpu',
            'sync_batchnorm': config.sync_batchnorm,
        }
    else:  # dp
        return {
            'strategy': 'dp',
            'devices': config.device_ids,
            'accelerator': 'gpu',
        }


# =============================================================================
# Registry
# =============================================================================

MULTI_GPU_COMPONENTS = {
    'detection': ['get_gpu_count', 'get_gpu_info', 'get_recommended_batch_size'],
    'dataparallel': ['wrap_data_parallel', 'unwrap_data_parallel'],
    'device': ['get_device', 'to_device', 'set_cuda_device'],
    'memory': ['get_memory_usage', 'clear_cuda_cache'],
    'config': ['MultiGPUConfig'],
    'lightning': ['get_lightning_strategy'],
}
