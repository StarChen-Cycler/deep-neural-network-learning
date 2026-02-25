"""Phase 1: Neural Network Basics."""

from .activations import (
    sigmoid,
    sigmoid_grad,
    tanh,
    tanh_grad,
    relu,
    relu_grad,
    leaky_relu,
    leaky_relu_grad,
    gelu,
    gelu_grad,
    swish,
    swish_grad,
    get_activation,
)

from .loss import (
    MSELoss,
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
    TripletLoss,
    get_loss,
)

from .mlp import (
    LinearLayer,
    ActivationLayer,
    MLP,
)

from .optimizer import (
    SGD,
    Momentum,
    Nesterov,
    AdaGrad,
    RMSprop,
    Adam,
    AdamW,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    get_optimizer,
    get_scheduler,
)

from .weight_init import (
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    kaiming_uniform,
    kaiming_normal,
    zero_init,
    lsuv_init,
    compute_fan,
    get_initializer,
    init_bias,
    INITIALIZERS,
)

__all__ = [
    # Activations
    "sigmoid",
    "sigmoid_grad",
    "tanh",
    "tanh_grad",
    "relu",
    "relu_grad",
    "leaky_relu",
    "leaky_relu_grad",
    "gelu",
    "gelu_grad",
    "swish",
    "swish_grad",
    "get_activation",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "TripletLoss",
    "get_loss",
    # MLP
    "LinearLayer",
    "ActivationLayer",
    "MLP",
    # Optimizers
    "SGD",
    "Momentum",
    "Nesterov",
    "AdaGrad",
    "RMSprop",
    "Adam",
    "AdamW",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "get_optimizer",
    "get_scheduler",
    # Weight initialization
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "zero_init",
    "lsuv_init",
    "compute_fan",
    "get_initializer",
    "init_bias",
    "INITIALIZERS",
]
