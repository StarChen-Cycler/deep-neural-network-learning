"""
Phase 5: Optimization & Deployment

This module provides optimization and deployment utilities:
    - Distributed training (DDP, DataParallel)
    - Multi-GPU configuration
    - Gradient accumulation
    - Memory optimization (checkpointing, offloading)
    - Model pruning and quantization
    - ONNX export
    - TensorRT optimization
    - Edge deployment

Modules:
    ddp_training: DDP setup, DistributedTrainer, gradient sync verification
    multi_gpu: GPU detection, DataParallel wrapper, memory management
    gradient_accumulation: Memory-efficient training with gradient accumulation
    memory_optimizer: Gradient checkpointing, CPU offloading, memory-efficient attention
    pruning: Model pruning (magnitude, channel, global, iterative)
    pruning_experiments: Compression benchmarks and fine-tuning pipelines
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
    get_memory_usage as multi_gpu_get_memory_usage,
    clear_cuda_cache,
    # Config
    MultiGPUConfig,
    # Lightning
    get_lightning_strategy,
    # Registry
    MULTI_GPU_COMPONENTS,
)

from .gradient_accumulation import (
    # Config
    GradientAccumulationConfig,
    # Core
    GradientAccumulator,
    # Trainer
    GradientAccumulationTrainer,
    # Utilities
    get_memory_usage,
    reset_memory_stats,
    benchmark_memory_usage,
    verify_gradient_equivalence,
    calculate_memory_savings,
    create_gradient_accumulation_trainer,
    recommend_accumulation_settings,
)

from .memory_optimizer import (
    # Config
    MemoryOptimizationConfig,
    # Memory Utilities
    get_memory_usage as optimizer_get_memory_usage,
    reset_memory_stats as optimizer_reset_memory_stats,
    get_peak_memory,
    clear_cuda_cache as optimizer_clear_cuda_cache,
    # Gradient Checkpointing
    get_checkpoint_segments,
    CheckpointedSequential,
    apply_gradient_checkpointing,
    # CPU Offloading
    CPUOffloader,
    OffloadedOptimizer,
    # Activation Recomputation
    ActivationRecomputer,
    # Memory Efficient Attention
    memory_efficient_attention,
    # In-place Operations
    enable_inplace_activation,
    # Trainer
    MemoryOptimizedTrainer,
    # Benchmarking
    benchmark_memory,
    compare_memory_strategies,
    # Registry
    MEMORY_OPTIMIZER_COMPONENTS,
)

from .pruning import (
    # Enums
    PruningMethod,
    PruningNorm,
    # Config
    PruningConfig,
    # Base
    BasePruner,
    # Pruners
    MagnitudePruner,
    RandomPruner,
    GradientPruner,
    ChannelPruner,
    GlobalPruner,
    # Schedule
    IterativePruningSchedule,
    # Manager
    PruningManager,
    # Utilities
    create_pruner,
    prune_model,
    get_model_sparsity,
    count_zero_weights,
    # Registry
    PRUNING_COMPONENTS,
)

from .pruning_experiments import (
    # Results
    SparsityResult,
    MethodResult,
    ExperimentReport,
    # Experiments
    SparsitySweepExperiment,
    MethodComparisonExperiment,
    FineTuningPipeline,
    CompletePruningExperiment,
    # Utilities
    evaluate_model,
    measure_inference_time,
    run_all_experiments,
    # Registry
    PRUNING_EXPERIMENTS_COMPONENTS,
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
    "multi_gpu_get_memory_usage",
    "clear_cuda_cache",
    # Multi-GPU Config
    "MultiGPUConfig",
    # Multi-GPU Lightning
    "get_lightning_strategy",
    # Multi-GPU Registry
    "MULTI_GPU_COMPONENTS",
    # Gradient Accumulation
    "GradientAccumulationConfig",
    "GradientAccumulator",
    "GradientAccumulationTrainer",
    "benchmark_memory_usage",
    "verify_gradient_equivalence",
    "calculate_memory_savings",
    "create_gradient_accumulation_trainer",
    "recommend_accumulation_settings",
    # Memory Optimization
    "MemoryOptimizationConfig",
    "get_memory_usage",
    "reset_memory_stats",
    "get_peak_memory",
    "get_checkpoint_segments",
    "CheckpointedSequential",
    "apply_gradient_checkpointing",
    "CPUOffloader",
    "OffloadedOptimizer",
    "ActivationRecomputer",
    "memory_efficient_attention",
    "enable_inplace_activation",
    "MemoryOptimizedTrainer",
    "benchmark_memory",
    "compare_memory_strategies",
    "MEMORY_OPTIMIZER_COMPONENTS",
    # Pruning
    "PruningMethod",
    "PruningNorm",
    "PruningConfig",
    "BasePruner",
    "MagnitudePruner",
    "RandomPruner",
    "GradientPruner",
    "ChannelPruner",
    "GlobalPruner",
    "IterativePruningSchedule",
    "PruningManager",
    "create_pruner",
    "prune_model",
    "get_model_sparsity",
    "count_zero_weights",
    "PRUNING_COMPONENTS",
    # Pruning Experiments
    "SparsityResult",
    "MethodResult",
    "ExperimentReport",
    "SparsitySweepExperiment",
    "MethodComparisonExperiment",
    "FineTuningPipeline",
    "CompletePruningExperiment",
    "evaluate_model",
    "measure_inference_time",
    "run_all_experiments",
    "PRUNING_EXPERIMENTS_COMPONENTS",
]
