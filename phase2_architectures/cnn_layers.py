"""
Convolutional Neural Network layers implemented from scratch with NumPy.

This module provides CNN core components with both forward and backward
propagation, using the im2col/col2im approach for efficient convolution.

Components:
    Conv2d: 2D convolution layer with forward and backward
    MaxPool2d: 2D max pooling with index tracking
    AvgPool2d: 2D average pooling
    Flatten: Flatten layer for transitioning to fully connected

Theory:
    Convolution Forward:
        output[b, c_out, h_out, w_out] = sum over (c_in, kh, kw) of
            input[b, c_in, h*stride + kh - pad, w*stride + kw - pad] * weight[c_out, c_in, kh, kw]

    Output Shape:
        H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

    Receptive Field:
        RF[n] = RF[n-1] + (kernel_size[n] - 1) * product_of_strides[0:n-1]

Implementation:
    Uses im2col (image to column) transformation for efficient matrix multiplication:
        1. Extract patches from input into columns
        2. Reshape weights to 2D matrix
        3. Perform single matrix multiplication
        4. Reshape result to output shape

References:
    - CS231n: Convolutional Neural Networks (Stanford)
    - Deep Learning (Goodfellow et al.): Chapter 9 - Convolutional Networks
"""

from typing import Optional, Tuple, Union, List
import numpy as np


def compute_output_shape(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    Compute output spatial dimension for convolution or pooling.

    Formula: out = floor((in + 2*pad - dil*(k-1) - 1) / stride + 1)

    Args:
        input_size: Input spatial dimension (H or W)
        kernel_size: Size of the kernel
        stride: Stride of the operation
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Returns:
        Output spatial dimension

    Examples:
        >>> compute_output_shape(32, 3, 1, 1)  # 3x3 conv, same padding
        32
        >>> compute_output_shape(32, 2, 2, 0)  # 2x2 maxpool
        16
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def compute_receptive_field(
    kernel_sizes: List[int],
    strides: List[int],
) -> int:
    """
    Compute receptive field of a stack of convolutional layers.

    Formula: RF[n] = RF[n-1] + (k[n] - 1) * product_of_strides[0:n-1]

    Args:
        kernel_sizes: List of kernel sizes for each layer
        strides: List of strides for each layer

    Returns:
        Total receptive field size

    Examples:
        >>> compute_receptive_field([3], [1])  # Single 3x3 conv
        3
        >>> compute_receptive_field([3, 3, 3], [1, 1, 1])  # Three 3x3 convs
        7
        >>> compute_receptive_field([7], [1])  # Single 7x7 conv
        7
    """
    rf = 1
    stride_product = 1

    for k, s in zip(kernel_sizes, strides):
        rf = rf + (k - 1) * stride_product
        stride_product *= s

    return rf


def im2col(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """
    Transform input tensor to column matrix for efficient convolution.

    This transformation allows convolution to be computed as a single
    matrix multiplication: output = col @ weight_reshaped

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Returns:
        Column matrix of shape (batch * out_h * out_w, channels * kernel_h * kernel_w)

    Examples:
        >>> x = np.random.randn(2, 3, 8, 8)  # batch=2, C=3, H=8, W=8
        >>> col = im2col(x, 3, 3, stride=1, padding=1)
        >>> col.shape
        (128, 27)  # 2*8*8=128 positions, 3*3*3=27 elements per patch
    """
    batch, channels, height, width = x.shape

    # Apply padding
    if padding > 0:
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_padded = x

    # Compute output dimensions
    out_h = compute_output_shape(height, kernel_h, stride, padding, dilation)
    out_w = compute_output_shape(width, kernel_w, stride, padding, dilation)

    # Effective kernel size with dilation
    eff_kh = dilation * (kernel_h - 1) + 1
    eff_kw = dilation * (kernel_w - 1) + 1

    # Create column matrix
    col = np.zeros((batch, channels, kernel_h, kernel_w, out_h, out_w))

    for kh in range(kernel_h):
        kh_dilated = kh * dilation
        for kw in range(kernel_w):
            kw_dilated = kw * dilation
            h_start = kh_dilated
            h_end = h_start + stride * out_h
            w_start = kw_dilated
            w_end = w_start + stride * out_w

            col[:, :, kh, kw, :, :] = x_padded[
                :, :, h_start:h_end:stride, w_start:w_end:stride
            ]

    # Reshape to (batch * out_h * out_w, channels * kernel_h * kernel_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1)

    return col


def col2im(
    col: np.ndarray,
    input_shape: Tuple[int, int, int, int],
    kernel_h: int,
    kernel_w: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """
    Transform column matrix back to image tensor (inverse of im2col).

    Used in backward pass to compute gradient w.r.t. input.

    Args:
        col: Column matrix of shape (batch * out_h * out_w, channels * kernel_h * kernel_w)
        input_shape: Original input shape (batch, channels, height, width)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Returns:
        Input tensor of shape (batch, channels, height, width)

    Note:
        When stride > 1, overlapping regions accumulate gradients.
    """
    batch, channels, height, width = input_shape

    # Compute output dimensions
    out_h = compute_output_shape(height, kernel_h, stride, padding, dilation)
    out_w = compute_output_shape(width, kernel_w, stride, padding, dilation)

    # Reshape column to (batch, out_h, out_w, channels, kernel_h, kernel_w)
    col = col.reshape(batch, out_h, out_w, channels, kernel_h, kernel_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    # Padded dimensions
    padded_h = height + 2 * padding
    padded_w = width + 2 * padding

    # Create padded output (accumulate overlapping regions)
    x_padded = np.zeros((batch, channels, padded_h, padded_w))

    for kh in range(kernel_h):
        kh_dilated = kh * dilation
        for kw in range(kernel_w):
            kw_dilated = kw * dilation
            h_start = kh_dilated
            h_end = h_start + stride * out_h
            w_start = kw_dilated
            w_end = w_start + stride * out_w

            x_padded[:, :, h_start:h_end:stride, w_start:w_end:stride] += col[
                :, :, kh, kw, :, :
            ]

    # Remove padding
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


def conv2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Forward pass for 2D convolution using im2col.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias of shape (out_channels,)
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Returns:
        Tuple of:
            - Output tensor of shape (batch, out_channels, out_height, out_width)
            - Column matrix (cached for backward)
            - Input shape (cached for backward)
    """
    batch, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output dimensions
    out_h = compute_output_shape(height, kernel_h, stride, padding, dilation)
    out_w = compute_output_shape(width, kernel_w, stride, padding, dilation)

    # im2col transformation
    col = im2col(x, kernel_h, kernel_w, stride, padding, dilation)

    # Reshape weight to (out_channels, in_channels * kernel_h * kernel_w)
    weight_col = weight.reshape(out_channels, -1)

    # Matrix multiplication: (batch * out_h * out_w, in_ch * kh * kw) @ (in_ch * kh * kw, out_ch)
    out = col @ weight_col.T  # Shape: (batch * out_h * out_w, out_channels)

    # Reshape to output tensor
    out = out.reshape(batch, out_h, out_w, out_channels).transpose(0, 3, 1, 2)

    # Add bias
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)

    return out, col, x.shape


def conv2d_backward(
    grad_output: np.ndarray,
    col: np.ndarray,
    weight: np.ndarray,
    input_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for 2D convolution.

    Args:
        grad_output: Gradient of loss w.r.t. output, shape (batch, out_channels, out_h, out_w)
        col: Cached column matrix from forward pass
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        input_shape: Original input shape
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Returns:
        Tuple of:
            - Gradient w.r.t. input, shape (batch, in_channels, height, width)
            - Gradient w.r.t. weight, shape (out_channels, in_channels, kernel_h, kernel_w)
            - Gradient w.r.t. bias, shape (out_channels,)
    """
    batch, out_channels, out_h, out_w = grad_output.shape
    _, in_channels, kernel_h, kernel_w = weight.shape

    # Reshape grad_output for matrix multiplication
    grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, out_channels)

    # Gradient w.r.t. bias: sum over batch and spatial dimensions
    grad_bias = np.sum(grad_output_reshaped, axis=0)

    # Gradient w.r.t. weight: col.T @ grad_output
    # col shape: (batch * out_h * out_w, in_ch * kh * kw)
    # grad_output_reshaped shape: (batch * out_h * out_w, out_ch)
    grad_weight = (col.T @ grad_output_reshaped).T.reshape(weight.shape)

    # Gradient w.r.t. input: grad_output @ weight, then col2im
    weight_col = weight.reshape(out_channels, -1)
    grad_col = grad_output_reshaped @ weight_col  # (batch * out_h * out_w, in_ch * kh * kw)

    # col2im transformation
    grad_input = col2im(grad_col, input_shape, kernel_h, kernel_w, stride, padding, dilation)

    return grad_input, grad_weight, grad_bias


class Conv2d:
    """
    2D Convolution layer implemented from scratch.

    Forward:
        output = conv2d(input, weight, bias)

    Backward:
        grad_input, grad_weight, grad_bias computed via im2col/col2im

    Attributes:
        weight: Convolution kernel of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias of shape (out_channels,)
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        dilation: Spacing between kernel elements

    Examples:
        >>> conv = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> x = np.random.randn(16, 3, 32, 32)  # batch=16, C=3, H=32, W=32
        >>> out = conv.forward(x)
        >>> out.shape
        (16, 64, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        """
        Initialize Conv2d layer with He initialization.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel (int or tuple)
            stride: Stride of the convolution
            padding: Zero-padding added to both sides
            dilation: Spacing between kernel elements
            bias: If True, adds a learnable bias
        """
        if isinstance(kernel_size, int):
            kernel_h, kernel_w = kernel_size, kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        # He initialization for ReLU networks
        # std = sqrt(2 / fan_in)
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        self.weight: np.ndarray = (
            np.random.randn(out_channels, in_channels, kernel_h, kernel_w) * std
        )
        self.bias: Optional[np.ndarray] = (
            np.zeros(out_channels) if bias else None
        )

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Gradients (computed in backward)
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Cache for backward pass
        self._col_cache: Optional[np.ndarray] = None
        self._input_shape_cache: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for 2D convolution.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width)
        """
        out, col, input_shape = conv2d_forward(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
        self._col_cache = col
        self._input_shape_cache = input_shape
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for 2D convolution.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._col_cache is None or self._input_shape_cache is None:
            raise RuntimeError("Must call forward before backward")

        grad_input, grad_weight, grad_bias = conv2d_backward(
            grad_output,
            self._col_cache,
            self.weight,
            self._input_shape_cache,
            self.stride,
            self.padding,
            self.dilation,
        )

        self.grad_weight = grad_weight
        self.grad_bias = grad_bias if self.bias is not None else None

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return list of gradients."""
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]
        return [self.grad_weight]

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        self.grad_weight = None
        self.grad_bias = None


class MaxPool2d:
    """
    2D Max Pooling layer with index tracking for backward pass.

    Forward:
        output[b, c, h, w] = max over (kh, kw) of
            input[b, c, h*stride + kh, w*stride + kw]

    Backward:
        Gradient routed only to the max element in each pool region.

    Attributes:
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (defaults to kernel_size)
        padding: Zero-padding (default 0)

    Examples:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = np.random.randn(16, 64, 32, 32)
        >>> out = pool.forward(x)
        >>> out.shape
        (16, 64, 16, 16)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
        padding: int = 0,
    ):
        """
        Initialize MaxPool2d layer.

        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling (defaults to kernel_size)
            padding: Zero-padding added to both sides
        """
        if isinstance(kernel_size, int):
            self.kernel_h, self.kernel_w = kernel_size, kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        self.stride = stride if stride is not None else self.kernel_h
        self.padding = padding

        # Cache for backward
        self._input_shape_cache: Optional[Tuple[int, ...]] = None
        self._max_indices_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for max pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, out_height, out_width)
        """
        batch, channels, height, width = x.shape
        self._input_shape_cache = x.shape

        out_h = compute_output_shape(height, self.kernel_h, self.stride, self.padding)
        out_w = compute_output_shape(width, self.kernel_w, self.stride, self.padding)

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
                constant_values=-np.inf,
            )
        else:
            x_padded = x

        # Initialize output and index storage
        output = np.zeros((batch, channels, out_h, out_w))
        max_indices = np.zeros((batch, channels, out_h, out_w, 2), dtype=np.int32)

        for h in range(out_h):
            for w in range(out_w):
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = h_start + self.kernel_h
                w_end = w_start + self.kernel_w

                # Get pooling region
                pool_region = x_padded[:, :, h_start:h_end, w_start:w_end]

                # Flatten for max
                pool_flat = pool_region.reshape(batch, channels, -1)
                max_idx = np.argmax(pool_flat, axis=2)

                # Convert flat index to 2D coordinates
                max_kh = max_idx // self.kernel_w
                max_kw = max_idx % self.kernel_w

                # Store indices (relative to original input, accounting for padding)
                max_indices[:, :, h, w, 0] = max_kh + h_start - self.padding
                max_indices[:, :, h, w, 1] = max_kw + w_start - self.padding

                # Get max values
                output[:, :, h, w] = np.max(pool_flat, axis=2)

        self._max_indices_cache = max_indices
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for max pooling.

        Gradient is routed only to the element that was the maximum
        in each pooling region during forward pass.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._input_shape_cache is None or self._max_indices_cache is None:
            raise RuntimeError("Must call forward before backward")

        batch, channels, height, width = self._input_shape_cache
        grad_input = np.zeros((batch, channels, height, width))

        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for h in range(out_h):
            for w in range(out_w):
                # Get the indices of max elements
                max_h = self._max_indices_cache[:, :, h, w, 0]
                max_w = self._max_indices_cache[:, :, h, w, 1]

                # Scatter gradients to max positions
                for b in range(batch):
                    for c in range(channels):
                        if 0 <= max_h[b, c] < height and 0 <= max_w[b, c] < width:
                            grad_input[b, c, max_h[b, c], max_w[b, c]] += grad_output[
                                b, c, h, w
                            ]

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return empty list (no learnable parameters)."""
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return empty list (no learnable parameters)."""
        return []

    def zero_grad(self) -> None:
        """No gradients to reset."""
        pass


class AvgPool2d:
    """
    2D Average Pooling layer.

    Forward:
        output[b, c, h, w] = mean over (kh, kw) of
            input[b, c, h*stride + kh, w*stride + kw]

    Backward:
        Gradient distributed equally to all elements in each pool region.

    Attributes:
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (defaults to kernel_size)
        padding: Zero-padding (default 0)

    Examples:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = np.random.randn(16, 64, 32, 32)
        >>> out = pool.forward(x)
        >>> out.shape
        (16, 64, 16, 16)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
        padding: int = 0,
    ):
        """
        Initialize AvgPool2d layer.

        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling (defaults to kernel_size)
            padding: Zero-padding added to both sides
        """
        if isinstance(kernel_size, int):
            self.kernel_h, self.kernel_w = kernel_size, kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        self.stride = stride if stride is not None else self.kernel_h
        self.padding = padding

        # Cache for backward
        self._input_shape_cache: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for average pooling.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, out_height, out_width)
        """
        batch, channels, height, width = x.shape
        self._input_shape_cache = x.shape

        out_h = compute_output_shape(height, self.kernel_h, self.stride, self.padding)
        out_w = compute_output_shape(width, self.kernel_w, self.stride, self.padding)

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x

        # Use im2col for efficient pooling
        col = im2col(x_padded, self.kernel_h, self.kernel_w, self.stride, 0)
        col = col.reshape(batch * out_h * out_w, channels, self.kernel_h * self.kernel_w)

        # Average pooling
        output = np.mean(col, axis=2)
        output = output.reshape(batch, out_h, out_w, channels).transpose(0, 3, 1, 2)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for average pooling.

        Gradient is distributed equally to all elements in each pool region.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if self._input_shape_cache is None:
            raise RuntimeError("Must call forward before backward")

        batch, channels, height, width = self._input_shape_cache
        grad_input = np.zeros((batch, channels, height, width))

        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        pool_size = self.kernel_h * self.kernel_w

        # Distribute gradient equally
        grad_per_element = grad_output / pool_size

        for h in range(out_h):
            for w in range(out_w):
                h_start = h * self.stride
                w_start = w * self.stride
                h_end = min(h_start + self.kernel_h, height)
                w_end = min(w_start + self.kernel_w, width)

                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_per_element[
                    :, :, h, w
                ][:, :, None, None]

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return empty list (no learnable parameters)."""
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return empty list (no learnable parameters)."""
        return []

    def zero_grad(self) -> None:
        """No gradients to reset."""
        pass


class Flatten:
    """
    Flatten layer for transitioning from convolutional to fully connected layers.

    Forward:
        output = input.reshape(batch, -1)

    Backward:
        grad_input = grad_output.reshape(input_shape)

    Examples:
        >>> flatten = Flatten()
        >>> x = np.random.randn(16, 64, 8, 8)  # Conv output
        >>> out = flatten.forward(x)
        >>> out.shape
        (16, 4096)
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        """
        Initialize Flatten layer.

        Args:
            start_dim: First dimension to flatten (default 1, preserve batch)
            end_dim: Last dimension to flatten (default -1, to the end)
        """
        self.start_dim = start_dim
        self.end_dim = end_dim

        # Cache for backward
        self._input_shape_cache: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for flatten.

        Args:
            x: Input tensor of shape (batch, ...)

        Returns:
            Flattened tensor of shape (batch, prod(dim[start_dim:end_dim]))
        """
        self._input_shape_cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for flatten.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input (reshaped to original shape)
        """
        if self._input_shape_cache is None:
            raise RuntimeError("Must call forward before backward")

        return grad_output.reshape(self._input_shape_cache)

    def parameters(self) -> List[np.ndarray]:
        """Return empty list (no learnable parameters)."""
        return []

    def gradients(self) -> List[Optional[np.ndarray]]:
        """Return empty list (no learnable parameters)."""
        return []

    def zero_grad(self) -> None:
        """No gradients to reset."""
        pass
