"""
Phase 3: Training Techniques

This module provides implementations of training techniques for deep learning:
    - Normalization: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
    - Dropout: Standard, MC, Variational, Alpha, Spatial, DropConnect
    - Regularization: L1, L2, ElasticNet, MaxNorm, SpectralNorm

Modules:
    normalization: 4 normalization techniques with forward/backward
    dropout: 6 dropout variants with train/eval modes
    regularization: Weight regularization methods
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
]
