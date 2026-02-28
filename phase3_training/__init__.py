"""
Phase 3: Training Techniques

This module provides implementations of training techniques for deep learning:
    - Normalization: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
    - Dropout: Standard, MC, Variational, Alpha, Spatial, DropConnect
    - Regularization: L1, L2, ElasticNet, MaxNorm, SpectralNorm
    - LR Schedulers: Step, Exponential, Cosine, Cyclic, OneCycle, etc.
    - Transfer Learning: Pretrained models, fine-tuning strategies
    - Fine-tuning: Strategy comparison, discriminative LRs

Modules:
    normalization: 4 normalization techniques with forward/backward
    dropout: 6 dropout variants with train/eval modes
    regularization: Weight regularization methods
    lr_scheduler: 11 learning rate scheduling strategies
    scheduler_comparison: Optimization comparison experiments
    transfer_learning: Transfer learning with pretrained models
    fine_tuning: Fine-tuning strategy comparison experiments
"""

from .normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    InstanceNorm2d,
    GroupNorm,
    NORMALIZATION_FUNCTIONS,
    get_normalization,
)

from .dropout import (
    Dropout,
    MCDropout,
    VariationalDropout,
    AlphaDropout,
    SpatialDropout,
    DropConnect,
    compute_mc_uncertainty,
    DROPOUT_FUNCTIONS,
    get_dropout,
)

from .regularization import (
    L1Regularization,
    L2Regularization,
    ElasticNet,
    L1L2Regularizer,
    MaxNormConstraint,
    OrthogonalRegularizer,
    SpectralNormConstraint,
    apply_weight_decay,
    compute_regularization_loss,
    REGULARIZATION_FUNCTIONS,
    get_regularizer,
)

from .lr_scheduler import (
    LRSchedulerBase,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LinearWarmup,
    CosineWarmup,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    WarmupDecayScheduler,
    CosineAnnealingWarmRestarts,
    PolynomialLR,
    get_scheduler,
    plot_learning_rate_curve,
    LR_SCHEDULERS,
)

from .transfer_learning import (
    TransferLearner,
    FineTuningStrategy,
    freeze_backbone,
    unfreeze_layers,
    get_discriminative_lr_params,
    create_layerwise_lr_groups,
    create_resnet50_transfer,
    train_with_transfer_learning,
    TRANSFER_LEARNING_FUNCTIONS,
)

from .fine_tuning import (
    create_synthetic_cifar10,
    create_realistic_synthetic_data,
    run_finetuning_experiment,
    compare_finetuning_strategies,
    run_discriminative_lr_ablation,
    run_convergence_analysis,
    generate_comparison_report,
    run_full_comparison_experiment,
    quick_test_transfer_learning,
)

__all__ = [
    # Normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "InstanceNorm2d",
    "GroupNorm",
    "NORMALIZATION_FUNCTIONS",
    "get_normalization",
    # Dropout
    "Dropout",
    "MCDropout",
    "VariationalDropout",
    "AlphaDropout",
    "SpatialDropout",
    "DropConnect",
    "compute_mc_uncertainty",
    "DROPOUT_FUNCTIONS",
    "get_dropout",
    # Regularization
    "L1Regularization",
    "L2Regularization",
    "ElasticNet",
    "L1L2Regularizer",
    "MaxNormConstraint",
    "OrthogonalRegularizer",
    "SpectralNormConstraint",
    "apply_weight_decay",
    "compute_regularization_loss",
    "REGULARIZATION_FUNCTIONS",
    "get_regularizer",
    # LR Schedulers
    "LRSchedulerBase",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "LinearWarmup",
    "CosineWarmup",
    "CyclicLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "WarmupDecayScheduler",
    "CosineAnnealingWarmRestarts",
    "PolynomialLR",
    "get_scheduler",
    "plot_learning_rate_curve",
    "LR_SCHEDULERS",
    # Transfer Learning
    "TransferLearner",
    "FineTuningStrategy",
    "freeze_backbone",
    "unfreeze_layers",
    "get_discriminative_lr_params",
    "create_layerwise_lr_groups",
    "create_resnet50_transfer",
    "train_with_transfer_learning",
    "TRANSFER_LEARNING_FUNCTIONS",
    # Fine-tuning
    "create_synthetic_cifar10",
    "create_realistic_synthetic_data",
    "run_finetuning_experiment",
    "compare_finetuning_strategies",
    "run_discriminative_lr_ablation",
    "run_convergence_analysis",
    "generate_comparison_report",
    "run_full_comparison_experiment",
    "quick_test_transfer_learning",
]
