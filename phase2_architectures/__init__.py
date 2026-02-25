"""
Phase 2: Neural Network Architectures

This module provides from-scratch implementations of common neural network
architectures including CNN, RNN/LSTM/GRU, and Transformer components.

Modules:
    cnn_layers: Convolutional and pooling layers (Conv2d, MaxPool2d, AvgPool2d)
    simple_cnn: Simple CNN and ResNet-like architectures for CIFAR10
    rnn_cells: RNN, LSTM, GRU implementations (future)
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
    # Utility functions
    "conv2d_forward",
    "conv2d_backward",
    "im2col",
    "col2im",
    "compute_output_shape",
    "compute_receptive_field",
    "count_parameters",
    "get_model_info",
]
