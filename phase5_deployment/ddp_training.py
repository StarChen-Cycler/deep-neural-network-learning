"""
Distributed Data Parallel (DDP) Training Implementation.

This module provides:
    - DDP setup and cleanup utilities
    - DistributedTrainer for multi-GPU training
    - Gradient synchronization verification
    - DataParallel vs DDP benchmark comparison
    - SyncBatchNorm conversion utilities

Theory:
    Distributed Data Parallel (DDP):
        - Each GPU has its own process with a replica of the model
        - Gradients are synchronized via allreduce across all processes
        - Uses bucket-based gradient reduction for efficiency
        - NCCL backend for optimal GPU communication

    DataParallel (DP) vs DDP:
        - DP: Single process, multi-threaded, GIL bottleneck
        - DDP: Multi-process, one per GPU, no GIL bottleneck
        - DDP is 30%+ faster due to true parallelism

    Gradient Synchronization:
        - DDP buckets gradients by size for efficient allreduce
        - Each bucket's gradients are averaged across all ranks
        - After backward(), all ranks have identical gradients

References:
    - DDP Tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    - DDP Notes: https://pytorch.org/docs/stable/notes/ddp.html
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import os
import sys
import time
import logging
import tempfile
from functools import partial

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    dist = None
    mp = None
    DDP = None
    DataLoader = None
    DistributedSampler = None
    autocast = None
    GradScaler = None

logger = logging.getLogger(__name__)


# =============================================================================
# DDP Setup and Cleanup
# =============================================================================


def is_ddp_available() -> bool:
    """
    Check if DDP is available.

    Returns:
        True if PyTorch and distributed are available
    """
    return HAS_TORCH and dist is not None


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU indices.

    Returns:
        List of GPU indices
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
    init_method: Optional[str] = None,
    master_addr: str = 'localhost',
    master_port: str = '29500',
    timeout_minutes: int = 30,
) -> None:
    """
    Initialize distributed process group for DDP.

    Args:
        rank: Rank of current process (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: URL for initialization (default: env://)
        master_addr: Master node address (for env initialization)
        master_port: Master node port (for env initialization)
        timeout_minutes: Timeout for initialization

    Raises:
        RuntimeError: If initialization fails
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for DDP")

    # Set environment variables for env:// initialization
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Set local rank for device assignment
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(rank)

    # Initialize process group
    if init_method is None:
        init_method = 'env://'

    try:
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

        logger.info(
            f"DDP initialized: rank={rank}, world_size={world_size}, "
            f"backend={backend}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize DDP: {e}")


def cleanup_ddp() -> None:
    """
    Clean up distributed process group.

    Should be called at the end of training.
    """
    if HAS_TORCH and dist is not None and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


def get_rank() -> int:
    """
    Get current process rank.

    Returns:
        Rank of current process, or 0 if not distributed
    """
    if HAS_TORCH and dist is not None and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get total number of processes.

    Returns:
        World size, or 1 if not distributed
    """
    if HAS_TORCH and dist is not None and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if rank 0 or not distributed
    """
    return get_rank() == 0


def barrier() -> None:
    """
    Synchronize all processes.

    Blocks until all processes reach this point.
    """
    if HAS_TORCH and dist is not None and dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(
    tensor: 'torch.Tensor',
    op: str = 'sum',
    average: bool = False,
) -> 'torch.Tensor':
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'max', 'min', 'product')
        average: Whether to average by world size

    Returns:
        Reduced tensor
    """
    if not HAS_TORCH or not dist.is_initialized():
        return tensor

    # Map operation names to dist ops
    ops = {
        'sum': dist.ReduceOp.SUM,
        'max': dist.ReduceOp.MAX,
        'min': dist.ReduceOp.MIN,
        'product': dist.ReduceOp.PRODUCT,
    }

    reduce_op = ops.get(op, dist.ReduceOp.SUM)

    # Clone to avoid modifying original
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=reduce_op)

    if average:
        tensor.div_(get_world_size())

    return tensor


# =============================================================================
# SyncBatchNorm Utilities
# =============================================================================


def convert_to_sync_batchnorm(model: 'nn.Module') -> 'nn.Module':
    """
    Convert all BatchNorm layers in model to SyncBatchNorm.

    SyncBatchNorm synchronizes batch statistics across all GPUs,
    which is important for accurate batch normalization in DDP.

    Args:
        model: PyTorch model with BatchNorm layers

    Returns:
        Model with SyncBatchNorm layers
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def is_sync_batchnorm(model: 'nn.Module') -> bool:
    """
    Check if model uses SyncBatchNorm.

    Args:
        model: PyTorch model

    Returns:
        True if any SyncBatchNorm layer is found
    """
    for module in model.modules():
        if isinstance(module, nn.SyncBatchNorm):
            return True
    return False


# =============================================================================
# Gradient Synchronization Verification
# =============================================================================


def verify_gradient_sync(
    model: 'nn.Module',
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify that gradients are synchronized across all processes.

    This function checks that gradients are identical (within tolerance)
    across all DDP processes after backward pass.

    Args:
        model: DDP-wrapped model
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (is_synced, error_dict) where error_dict contains
        max gradient error for each parameter
    """
    if not HAS_TORCH or not dist.is_initialized():
        return True, {}

    errors = {}
    all_synced = True

    # Get the underlying module if wrapped in DDP
    if isinstance(model, DDP):
        model = model.module

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Gather gradients from all ranks
        grad = param.grad.detach().clone()

        # Use all_reduce to check if all gradients are identical
        # If all are same, max should equal any individual value
        grad_max = grad.clone()
        dist.all_reduce(grad_max, op=dist.ReduceOp.MAX)

        grad_min = grad.clone()
        dist.all_reduce(grad_min, op=dist.ReduceOp.MIN)

        # Check if max == min (all values same)
        error = (grad_max - grad_min).abs().max().item()
        errors[name] = error

        if error > rtol:
            all_synced = False
            if is_main_process():
                logger.warning(f"Gradient sync error for {name}: {error}")

    return all_synced, errors


def get_gradient_sync_error(model: 'nn.Module') -> float:
    """
    Get maximum gradient synchronization error across all parameters.

    Args:
        model: DDP-wrapped model

    Returns:
        Maximum gradient error across all parameters
    """
    _, errors = verify_gradient_sync(model)
    return max(errors.values()) if errors else 0.0


# =============================================================================
# Distributed Sampler
# =============================================================================


def create_distributed_sampler(
    dataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> 'DistributedSampler':
    """
    Create a DistributedSampler for DDP training.

    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes (default: world_size)
        rank: Rank of current process (default: current rank)
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        drop_last: Whether to drop last incomplete batch

    Returns:
        DistributedSampler instance
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    if num_replicas is None:
        num_replicas = get_world_size()
    if rank is None:
        rank = get_rank()

    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> 'DataLoader':
    """
    Create a DataLoader with DistributedSampler for DDP.

    Note: shuffle in DataLoader is ignored when using DistributedSampler.
    Use sampler.shuffle or set_epoch() instead.

    Args:
        dataset: Dataset to load
        batch_size: Per-GPU batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader with DistributedSampler
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    sampler = create_distributed_sampler(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# =============================================================================
# Distributed Trainer
# =============================================================================


@dataclass
class DDPConfig:
    """
    Configuration for DDP training.

    Attributes:
        backend: Communication backend ('nccl', 'gloo')
        master_addr: Master node address
        master_port: Master node port
        sync_batchnorm: Whether to use SyncBatchNorm
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Use gradient as bucket view for memory efficiency
        broadcast_buffers: Whether to broadcast buffers
        bucket_cap_mb: Bucket capacity in MB
    """
    backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: str = '29500'
    sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


class DistributedTrainer:
    """
    Trainer for Distributed Data Parallel training.

    Features:
        - Automatic DDP setup and cleanup
        - Gradient synchronization verification
        - Mixed precision support
        - DistributedSampler handling
        - Checkpoint saving from main process only
        - Logging from main process only

    Usage:
        def train_fn(rank, world_size, config):
            trainer = DistributedTrainer(
                model_class=MyModel,
                train_dataset=train_data,
                config=config,
                rank=rank,
                world_size=world_size,
            )
            trainer.train(epochs=10)

        mp.spawn(train_fn, args=(world_size, config), nprocs=world_size)
    """

    def __init__(
        self,
        model: Union['nn.Module', type],
        optimizer_class: type,
        criterion: Callable,
        train_dataset,
        val_dataset=None,
        config: Optional[DDPConfig] = None,
        rank: int = 0,
        world_size: int = 1,
        lr: float = 0.001,
        batch_size: int = 32,
        num_workers: int = 0,
        grad_clip_norm: Optional[float] = None,
        use_amp: bool = True,
        mixed_precision: str = 'auto',
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the distributed trainer.

        Args:
            model: Model instance or class
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            criterion: Loss function
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: DDP configuration
            rank: Current process rank
            world_size: Total number of processes
            lr: Learning rate
            batch_size: Per-GPU batch size
            num_workers: Number of data loading workers
            grad_clip_norm: Optional gradient clipping norm
            use_amp: Whether to use automatic mixed precision
            mixed_precision: Precision mode ('fp16', 'bf16', 'fp32', 'auto')
            model_kwargs: Additional model constructor arguments
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for DistributedTrainer")

        self.rank = rank
        self.world_size = world_size
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        self.mixed_precision = mixed_precision
        self.criterion = criterion

        # Set device
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)

        # Setup DDP
        if config is None:
            config = DDPConfig()

        self.config = config
        setup_ddp(
            rank=rank,
            world_size=world_size,
            backend=config.backend,
            master_addr=config.master_addr,
            master_port=config.master_port,
        )

        # Create model
        if isinstance(model, type):
            model_kwargs = model_kwargs or {}
            self.model = model(**model_kwargs)
        else:
            self.model = model

        # Convert to SyncBatchNorm if requested
        if config.sync_batchnorm:
            self.model = convert_to_sync_batchnorm(self.model)

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            broadcast_buffers=config.broadcast_buffers,
            bucket_cap_mb=config.bucket_cap_mb,
        )

        # Create optimizer
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)

        # Create dataloaders
        self.train_loader = create_distributed_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = create_distributed_dataloader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
            )

        # Setup AMP
        if use_amp:
            from ..phase4_advanced.mixed_precision import (
                is_bf16_supported,
                is_fp16_supported,
            )
            if mixed_precision == 'auto':
                if is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                elif is_fp16_supported():
                    self.amp_dtype = torch.float16
                else:
                    self.amp_dtype = torch.float32
                    self.use_amp = False
            elif mixed_precision == 'bf16':
                self.amp_dtype = torch.bfloat16
            elif mixed_precision == 'fp16':
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float32
                self.use_amp = False

            if self.use_amp:
                self.scaler = GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        # Training statistics
        self._step_count = 0
        self._epoch_count = 0
        self._best_loss = float('inf')
        self._loss_history: List[float] = []

    def train_step(
        self,
        inputs: 'torch.Tensor',
        targets: 'torch.Tensor',
    ) -> float:
        """
        Perform a single training step.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if self.use_amp:
            with autocast(device_type='cuda', dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.optimizer.step()

        self._step_count += 1
        return loss.item()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()

        # Set epoch for DistributedSampler shuffling
        self.train_loader.sampler.set_epoch(self._epoch_count)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            loss = self.train_step(inputs, targets)
            epoch_losses.append(loss)

        # Synchronize before computing statistics
        barrier()

        # Average loss across all processes
        avg_loss_tensor = torch.tensor(
            [sum(epoch_losses) / len(epoch_losses)],
            device=self.device
        )
        avg_loss_tensor = all_reduce_tensor(avg_loss_tensor, average=True)
        avg_loss = avg_loss_tensor.item()

        epoch_time = time.time() - epoch_start
        self._epoch_count += 1

        if avg_loss < self._best_loss:
            self._best_loss = avg_loss

        self._loss_history.append(avg_loss)

        return {
            'epoch': self._epoch_count,
            'avg_loss': avg_loss,
            'best_loss': self._best_loss,
            'epoch_time': epoch_time,
            'step_count': self._step_count,
        }

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary with validation statistics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                val_losses.append(loss.item())

                # Accuracy calculation (for classification)
                if targets.dim() == 1:  # Classification labels
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

        barrier()

        # Average across all processes
        avg_loss_tensor = torch.tensor(
            [sum(val_losses) / len(val_losses)],
            device=self.device
        )
        avg_loss_tensor = all_reduce_tensor(avg_loss_tensor, average=True)
        avg_loss = avg_loss_tensor.item()

        metrics = {'val_loss': avg_loss}

        if total > 0:
            accuracy_tensor = torch.tensor([correct / total], device=self.device)
            accuracy_tensor = all_reduce_tensor(accuracy_tensor, average=True)
            metrics['val_accuracy'] = accuracy_tensor.item()

        return metrics

    def train(
        self,
        epochs: int,
        validate_every: int = 1,
        log_every: int = 10,
        save_best: bool = True,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            epochs: Number of epochs to train
            validate_every: Run validation every N epochs
            log_every: Log every N epochs (main process only)
            save_best: Save best model checkpoint
            checkpoint_dir: Directory for checkpoints

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_times': [],
        }

        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['avg_loss'])
            history['epoch_times'].append(train_metrics['epoch_time'])

            # Validate
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                history['val_loss'].append(val_metrics.get('val_loss', 0))
                history['val_accuracy'].append(val_metrics.get('val_accuracy', 0))

            # Log (main process only)
            if is_main_process() and (epoch + 1) % log_every == 0:
                msg = f"Epoch {epoch+1}/{epochs}: loss={train_metrics['avg_loss']:.4f}"
                if 'val_loss' in val_metrics if 'val_metrics' in dir() else {}:
                    msg += f", val_loss={val_metrics['val_loss']:.4f}"
                logger.info(msg)

            # Save checkpoint
            if save_best and checkpoint_dir and is_main_process():
                if train_metrics['avg_loss'] <= self._best_loss:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, 'best_model.pt'),
                        epoch=epoch,
                    )

        return history

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Save model checkpoint (main process only).

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
        """
        if not is_main_process():
            return

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch or self._epoch_count,
            'best_loss': self._best_loss,
            'loss_history': self._loss_history,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._epoch_count = checkpoint.get('epoch', 0)
        self._best_loss = checkpoint.get('best_loss', float('inf'))
        self._loss_history = checkpoint.get('loss_history', [])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Checkpoint loaded from {path}")

    def cleanup(self) -> None:
        """Clean up DDP resources."""
        cleanup_ddp()

    def verify_gradients(self) -> Tuple[bool, float]:
        """
        Verify gradient synchronization.

        Returns:
            Tuple of (is_synced, max_error)
        """
        is_synced, errors = verify_gradient_sync(self.model)
        max_error = max(errors.values()) if errors else 0.0
        return is_synced, max_error


# =============================================================================
# DataParallel vs DDP Benchmark
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from DP vs DDP benchmark."""
    dp_time: float
    ddp_time: float
    speedup: float
    dp_memory_mb: float
    ddp_memory_mb: float
    gradient_sync_error: float
    meets_criteria: bool  # DDP 30%+ faster than DP


def benchmark_dp_vs_ddp(
    model_class: type,
    dataset,
    batch_size: int = 32,
    num_steps: int = 100,
    criterion: Optional[Callable] = None,
    model_kwargs: Optional[Dict] = None,
) -> BenchmarkResult:
    """
    Benchmark DataParallel vs DDP training speed.

    This function runs a comparison benchmark on a single node with
    multiple GPUs.

    Args:
        model_class: Model class to benchmark
        dataset: Dataset for training
        batch_size: Per-GPU batch size
        num_steps: Number of steps to benchmark
        criterion: Loss function (default: MSELoss)
        model_kwargs: Additional model arguments

    Returns:
        BenchmarkResult with timing and memory statistics

    Note:
        This function spawns multiple processes for DDP benchmark.
        Must be called from main module.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    if criterion is None:
        criterion = nn.MSELoss()

    model_kwargs = model_kwargs or {}
    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        raise RuntimeError("At least 2 GPUs required for DP vs DDP benchmark")

    # ===================
    # DataParallel Benchmark
    # ===================
    model_dp = model_class(**model_kwargs)
    model_dp = nn.DataParallel(model_dp)
    model_dp = model_dp.cuda()

    optimizer_dp = torch.optim.SGD(model_dp.parameters(), lr=0.001)

    dataloader_dp = DataLoader(
        dataset,
        batch_size=batch_size * num_gpus,  # DP splits batch across GPUs
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    data_iter = iter(dataloader_dp)
    dp_times = []
    dp_memory = []

    torch.cuda.reset_peak_memory_stats()

    for step in range(num_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader_dp)
            inputs, targets = next(data_iter)

        inputs = inputs.cuda()
        targets = targets.cuda()

        start_time = time.time()
        optimizer_dp.zero_grad()
        outputs = model_dp(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_dp.step()
        dp_times.append(time.time() - start_time)
        dp_memory.append(torch.cuda.max_memory_allocated() / (1024**2))

    dp_avg_time = sum(dp_times) / len(dp_times)
    dp_max_memory = max(dp_memory)

    # Clear memory
    del model_dp, optimizer_dp, dataloader_dp, data_iter
    torch.cuda.empty_cache()

    # ===================
    # DDP Benchmark (spawn processes)
    # ===================
    ddp_times_list = []
    ddp_memory_list = []
    gradient_error = 0.0

    def ddp_benchmark_fn(rank: int, world_size: int, results_queue):
        """DDP benchmark function for each process."""
        try:
            setup_ddp(rank=rank, world_size=world_size, backend='nccl')

            device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(device)

            # Create model
            model = model_class(**model_kwargs)
            model = convert_to_sync_batchnorm(model)
            model = model.to(device)
            model = DDP(model, device_ids=[rank])

            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            # Create dataloader
            sampler = DistributedSampler(dataset, shuffle=True)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
            )

            criterion_local = nn.MSELoss() if criterion is None else criterion

            torch.cuda.reset_peak_memory_stats()
            times = []
            memory = []
            grad_errors = []

            for step in range(num_steps):
                sampler.set_epoch(step)

                try:
                    idx = step % len(dataloader)
                    inputs, targets = list(dataloader)[idx]
                except:
                    inputs, targets = next(iter(dataloader))

                inputs = inputs.to(device)
                targets = targets.to(device)

                start_time = time.time()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion_local(outputs, targets)
                loss.backward()

                # Check gradient sync
                is_synced, errors = verify_gradient_sync(model)
                max_error = max(errors.values()) if errors else 0.0
                grad_errors.append(max_error)

                optimizer.step()
                times.append(time.time() - start_time)
                memory.append(torch.cuda.max_memory_allocated() / (1024**2))

            if rank == 0:
                results_queue.put({
                    'times': times,
                    'memory': memory,
                    'grad_error': max(grad_errors),
                })

            cleanup_ddp()

        except Exception as e:
            if rank == 0:
                results_queue.put({'error': str(e)})

    # Run DDP benchmark
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=ddp_benchmark_fn,
            args=(rank, num_gpus, results_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Get results from rank 0
    if not results_queue.empty():
        ddp_results = results_queue.get()
        if 'error' in ddp_results:
            raise RuntimeError(f"DDP benchmark failed: {ddp_results['error']}")
        ddp_times_list = ddp_results['times']
        ddp_memory_list = ddp_results['memory']
        gradient_error = ddp_results['grad_error']

    ddp_avg_time = sum(ddp_times_list) / len(ddp_times_list) if ddp_times_list else 0
    ddp_max_memory = max(ddp_memory_list) if ddp_memory_list else 0

    # Calculate speedup
    speedup = (dp_avg_time - ddp_avg_time) / dp_avg_time * 100  # Percentage

    return BenchmarkResult(
        dp_time=dp_avg_time,
        ddp_time=ddp_avg_time,
        speedup=speedup,
        dp_memory_mb=dp_max_memory,
        ddp_memory_mb=ddp_max_memory,
        gradient_sync_error=gradient_error,
        meets_criteria=speedup >= 30.0 and gradient_error < 1e-6,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def spawn_ddp_training(
    train_fn: Callable,
    world_size: Optional[int] = None,
    args: tuple = (),
) -> None:
    """
    Spawn DDP training processes.

    Args:
        train_fn: Training function (takes rank, world_size as first args)
        world_size: Number of processes (default: number of GPUs)
        args: Additional arguments for train_fn
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    if world_size is None:
        world_size = torch.cuda.device_count()

    if world_size < 1:
        raise RuntimeError("No GPUs available for DDP training")

    mp.spawn(
        train_fn,
        args=(world_size,) + args,
        nprocs=world_size,
        join=True,
    )


def get_ddp_info() -> Dict[str, Any]:
    """
    Get DDP environment information.

    Returns:
        Dictionary with DDP info
    """
    info = {
        'ddp_available': is_ddp_available(),
        'cuda_available': False,
        'num_gpus': 0,
        'gpu_names': [],
        'is_distributed': False,
        'rank': 0,
        'world_size': 1,
    }

    if not HAS_TORCH:
        return info

    info['cuda_available'] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        info['gpu_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(info['num_gpus'])
        ]

    if dist is not None and dist.is_initialized():
        info['is_distributed'] = True
        info['rank'] = dist.get_rank()
        info['world_size'] = dist.get_world_size()

    return info


# =============================================================================
# Registry
# =============================================================================

DDP_TRAINING_COMPONENTS = {
    'setup': ['setup_ddp', 'cleanup_ddp', 'is_ddp_available'],
    'info': ['get_rank', 'get_world_size', 'is_main_process', 'barrier', 'get_ddp_info'],
    'communication': ['all_reduce_tensor'],
    'batchnorm': ['convert_to_sync_batchnorm', 'is_sync_batchnorm'],
    'gradient': ['verify_gradient_sync', 'get_gradient_sync_error'],
    'sampler': ['create_distributed_sampler', 'create_distributed_dataloader'],
    'trainer': ['DistributedTrainer', 'DDPConfig'],
    'benchmark': ['benchmark_dp_vs_ddp', 'BenchmarkResult'],
    'spawn': ['spawn_ddp_training'],
}
