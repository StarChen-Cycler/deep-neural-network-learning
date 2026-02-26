"""
Phase 2: Neural Network Architectures

This module provides from-scratch implementations of common neural network
architectures including CNN, RNN/LSTM/GRU, and Transformer components.

Modules:
    cnn_layers: Convolutional and pooling layers (Conv2d, MaxPool2d, AvgPool2d)
    simple_cnn: Simple CNN and ResNet-like architectures for CIFAR10
    rnn_cells: RNN, LSTM, GRU implementations
    attention: Self-attention and multi-head attention (future)
"""

from .cnn_layers import (
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    Flatten,
    conv2d_forward,
    conv2d_backward,
    im2col,
    col2im,
    compute_output_shape,
    compute_receptive_field,
)

from .simple_cnn import (
    BatchNorm2d,
    ReLULayer,
    SimpleCNN,
    ResidualBlock,
    ResNetSmall,
    count_parameters,
    get_model_info,
)

from .rnn_cells import (
    RNNCell,
    LSTMCell,
    GRUCell,
    RNN,
    LSTM,
    GRU,
    count_parameters_rnn,
    gradient_clip,
    get_rnn_cell,
    get_rnn_model,
)

from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerEncoderLayer,
    create_causal_mask,
    create_padding_mask,
    count_parameters_attention,
)

__all__ = [
    # CNN Layers
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "Flatten",
    # CNN Models
    "BatchNorm2d",
    "ReLULayer",
    "SimpleCNN",
    "ResidualBlock",
    "ResNetSmall",
    # RNN Cells
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    # RNN Models
    "RNN",
    "LSTM",
    "GRU",
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SinusoidalPositionalEncoding",
    "TransformerEncoderLayer",
    # Utility functions
    "conv2d_forward",
    "conv2d_backward",
    "im2col",
    "col2im",
    "compute_output_shape",
    "compute_receptive_field",
    "count_parameters",
    "get_model_info",
    "count_parameters_rnn",
    "gradient_clip",
    "get_rnn_cell",
    "get_rnn_model",
    "create_causal_mask",
    "create_padding_mask",
    "count_parameters_attention",
]
