"""
Phase 5: Optimization & Deployment

This module provides optimization and deployment utilities:
    - Distributed training (DDP, DataParallel)
    - Multi-GPU configuration
    - Model pruning and quantization
    - ONNX export
    - TensorRT optimization
    - Edge deployment

Modules:
    ddp_training: DDP setup, DistributedTrainer, gradient sync verification
    multi_gpu: GPU detection, DataParallel wrapper, memory management
"""

from .ddp_training import (
    # Setup
    setup_ddp,
    cleanup_ddp,
    is_ddp_available,
    # Info
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    get_ddp_info,
    # Communication
    all_reduce_tensor,
    # BatchNorm
    convert_to_sync_batchnorm,
    is_sync_batchnorm,
    # Gradient
    verify_gradient_sync,
    get_gradient_sync_error,
    # Sampler
    create_distributed_sampler,
    create_distributed_dataloader,
    # Trainer
    DistributedTrainer,
    DDPConfig,
    # Benchmark
    benchmark_dp_vs_ddp,
    BenchmarkResult,
    # Spawn
    spawn_ddp_training,
    # Registry
    DDP_TRAINING_COMPONENTS,
)

from .multi_gpu import (
    # Detection
    get_gpu_count,
    get_gpu_info,
    get_recommended_batch_size,
    # DataParallel
    wrap_data_parallel,
    unwrap_data_parallel,
    # Device
    get_device,
    to_device,
    set_cuda_device,
    # Memory
    get_memory_usage,
    clear_cuda_cache,
    # Config
    MultiGPUConfig,
    # Lightning
    get_lightning_strategy,
    # Registry
    MULTI_GPU_COMPONENTS,
)

__all__ = [
    # DDP Setup
    "setup_ddp",
    "cleanup_ddp",
    "is_ddp_available",
    # DDP Info
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "get_ddp_info",
    # DDP Communication
    "all_reduce_tensor",
    # DDP BatchNorm
    "convert_to_sync_batchnorm",
    "is_sync_batchnorm",
    # DDP Gradient
    "verify_gradient_sync",
    "get_gradient_sync_error",
    # DDP Sampler
    "create_distributed_sampler",
    "create_distributed_dataloader",
    # DDP Trainer
    "DistributedTrainer",
    "DDPConfig",
    # DDP Benchmark
    "benchmark_dp_vs_ddp",
    "BenchmarkResult",
    # DDP Spawn
    "spawn_ddp_training",
    # DDP Registry
    "DDP_TRAINING_COMPONENTS",
    # Multi-GPU Detection
    "get_gpu_count",
    "get_gpu_info",
    "get_recommended_batch_size",
    # Multi-GPU DataParallel
    "wrap_data_parallel",
    "unwrap_data_parallel",
    # Multi-GPU Device
    "get_device",
    "to_device",
    "set_cuda_device",
    # Multi-GPU Memory
    "get_memory_usage",
    "clear_cuda_cache",
    # Multi-GPU Config
    "MultiGPUConfig",
    # Multi-GPU Lightning
    "get_lightning_strategy",
    # Multi-GPU Registry
    "MULTI_GPU_COMPONENTS",
]
