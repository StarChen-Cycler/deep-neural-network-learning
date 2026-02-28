"""
Model Quantization Implementation for Neural Network Compression.

This module provides:
    - DynamicQuantizer: Weight quantization at runtime
    - StaticQuantizer: PTQ with calibration data
    - QATQuantizer: Quantization-Aware Training
    - INT8/INT4 quantization support
    - ONNX export for quantized models

Theory:
    Quantization reduces model size and improves inference speed by converting
    floating-point weights and activations to lower precision integers.

    Dynamic Quantization:
        - Weights quantized ahead of time
        - Activations quantized dynamically at runtime
        - No calibration data needed
        - Best for: LSTM, Transformer models

    Static Quantization (PTQ):
        - Both weights and activations quantized ahead of time
        - Requires calibration data to determine activation ranges
        - Fastest inference
        - Best for: CNN models

    Quantization-Aware Training (QAT):
        - Simulates quantization during training
        - Model learns to compensate for quantization error
        - Best accuracy retention
        - Best for: When accuracy is critical

    Quantization Formula:
        q = round(r / s) + z
        where:
            q = quantized integer value
            r = real floating-point value
            s = scale factor
            z = zero point

References:
    - PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
    - Quantization Whitepaper: https://arxiv.org/abs/1806.08342
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
import time
import logging
import os
import warnings

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.ao.quantization import (
        get_default_qconfig,
        get_default_qat_qconfig,
        QConfig,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        HistogramObserver,
        PerChannelMinMaxObserver,
    )
    from torch.ao.quantization.quantize import (
        prepare,
        prepare_qat,
        convert,
    )
    from torch.ao.quantization.quantize_fx import (
        prepare_fx,
        prepare_qat_fx,
        convert_fx,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class QuantizationType(Enum):
    """Supported quantization types."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization-Aware Training


class QuantizationDtype(Enum):
    """Quantization data types."""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bfloat16"


class ObserverType(Enum):
    """Observer types for calibration."""
    MINMAX = "minmax"
    MOVING_AVERAGE = "moving_average"
    HISTOGRAM = "histogram"
    PER_CHANNEL = "per_channel"


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Attributes:
        qtype: Type of quantization (dynamic, static, qat)
        dtype: Target data type (int8, int4, fp16)
        observer: Observer type for calibration
        per_channel: Use per-channel quantization
        fuse_modules: List of module patterns to fuse
        calibration_batches: Number of batches for calibration
        qat_epochs: Epochs for QAT fine-tuning
        qat_learning_rate: Learning rate for QAT
        layers_to_quantize: Specific layers to quantize (None = all)
        backend: Quantization backend (x86, qnnpack, etc.)
    """
    qtype: QuantizationType = QuantizationType.STATIC
    dtype: QuantizationDtype = QuantizationDtype.INT8
    observer: ObserverType = ObserverType.HISTOGRAM
    per_channel: bool = True
    fuse_modules: Optional[List[List[str]]] = None
    calibration_batches: int = 10
    qat_epochs: int = 5
    qat_learning_rate: float = 1e-5
    layers_to_quantize: Optional[List[str]] = None
    backend: str = "x86"

    def __post_init__(self):
        """Validate configuration."""
        if self.qtype == QuantizationType.STATIC and self.calibration_batches < 1:
            raise ValueError("Static quantization requires at least 1 calibration batch")
        if self.qtype == QuantizationType.QAT and self.qat_epochs < 1:
            raise ValueError("QAT requires at least 1 epoch")


# =============================================================================
# Base Quantizer Class
# =============================================================================


class BaseQuantizer:
    """
    Base class for all quantization methods.

    Provides common utilities for:
        - Model size calculation
        - Inference time measurement
        - Accuracy evaluation
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
        """
        if config is None:
            config = QuantizationConfig()
        self.config = config
        self._original_model: Optional["nn.Module"] = None

    def save_original_model(self, model: "nn.Module") -> None:
        """Save a copy of the original model."""
        self._original_model = copy.deepcopy(model)

    def get_model_size_mb(self, model: "nn.Module") -> float:
        """Get model size in megabytes."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return (param_size + buffer_size) / (1024 * 1024)

    def count_parameters(self, model: "nn.Module") -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())

    def measure_inference_time(
        self,
        model: "nn.Module",
        input_shape: Tuple[int, ...],
        device: str = 'cpu',
        n_runs: int = 100,
        warmup: int = 10
    ) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()
        model.to(device)

        dummy_input = torch.randn(1, *input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        end = time.perf_counter()

        return (end - start) / n_runs * 1000

    def get_qconfig(self) -> "QConfig":
        """Get PyTorch QConfig based on configuration."""
        if self.config.dtype == QuantizationDtype.INT8:
            if self.config.observer == ObserverType.MINMAX:
                observer = MinMaxObserver
            elif self.config.observer == ObserverType.MOVING_AVERAGE:
                observer = MovingAverageMinMaxObserver
            elif self.config.observer == ObserverType.PER_CHANNEL:
                observer = PerChannelMinMaxObserver
            else:
                observer = HistogramObserver

            if self.config.per_channel:
                weight_observer = PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                )
            else:
                weight_observer = MinMaxObserver.with_args(dtype=torch.qint8)

            activation_observer = observer.with_args(dtype=torch.quint8)

            return QConfig(activation=activation_observer, weight=weight_observer)
        else:
            # Default QConfig for other dtypes
            return get_default_qconfig(self.config.backend)


# =============================================================================
# Dynamic Quantizer
# =============================================================================


class DynamicQuantizer(BaseQuantizer):
    """
    Dynamic quantization - weights quantized ahead, activations at runtime.

    Benefits:
        - No calibration data needed
        - Easy to apply
        - Good for LSTM/Transformer models

    Limitations:
        - Less inference speedup than static quantization
        - Only weight quantization is static
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__(config)

    def quantize(
        self,
        model: "nn.Module",
        dtype: Optional[QuantizationDtype] = None
    ) -> "nn.Module":
        """
        Apply dynamic quantization.

        Args:
            model: PyTorch model
            dtype: Target dtype (default: INT8)

        Returns:
            Quantized model
        """
        if dtype is None:
            dtype = self.config.dtype

        model.eval()

        # Determine quantized dtype
        if dtype == QuantizationDtype.INT8:
            q_dtype = torch.qint8
        elif dtype == QuantizationDtype.INT4:
            q_dtype = torch.qint8  # INT4 not directly supported, use INT8
        else:
            q_dtype = torch.float16

        # Get layer types to quantize
        q_spec = {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}
        if self.config.layers_to_quantize:
            # Filter to specified layers
            pass

        # Apply dynamic quantization
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            q_spec,
            dtype=q_dtype
        )

        logger.info(f"Dynamic quantization applied with dtype={dtype.value}")
        return quantized_model


# =============================================================================
# Static Quantizer (PTQ)
# =============================================================================


class StaticQuantizer(BaseQuantizer):
    """
    Static quantization (Post-Training Quantization).

    Both weights and activations quantized ahead of time.
    Requires calibration data to determine activation ranges.

    Benefits:
        - Fastest inference
        - Best for CNN models
        - No retraining needed

    Limitations:
        - Requires representative calibration data
        - May have accuracy drop for some models
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__(config)

    def _fuse_modules(self, model: "nn.Module") -> "nn.Module":
        """Fuse modules for better quantization."""
        if self.config.fuse_modules:
            for fuse_list in self.config.fuse_modules:
                try:
                    torch.ao.quantization.fuse_modules(model, fuse_list, inplace=True)
                except Exception as e:
                    logger.warning(f"Could not fuse {fuse_list}: {e}")
        return model

    def calibrate(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Run calibration to determine activation ranges.

        Args:
            model: Prepared model with observers
            dataloader: Calibration data
            device: Device to use

        Returns:
            Calibrated model
        """
        model.eval()
        model.to(device)

        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= self.config.calibration_batches:
                    break
                inputs = inputs.to(device)
                model(inputs)

        logger.info(f"Calibration completed with {self.config.calibration_batches} batches")
        return model

    def quantize(
        self,
        model: "nn.Module",
        calibration_loader: Optional["DataLoader"] = None,
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Apply static quantization.

        Args:
            model: PyTorch model
            calibration_loader: DataLoader with calibration data
            device: Device to use

        Returns:
            Quantized model
        """
        model.eval()

        # Create a copy to avoid modifying original
        model = copy.deepcopy(model)

        # Fuse modules
        model = self._fuse_modules(model)

        # Set qconfig
        qconfig = self.get_qconfig()
        model.qconfig = qconfig

        # Prepare for quantization
        torch.ao.quantization.prepare(model, inplace=True)

        # Calibrate if data provided
        if calibration_loader is not None:
            model = self.calibrate(model, calibration_loader, device)

        # Convert to quantized model
        torch.ao.quantization.convert(model, inplace=True)

        logger.info("Static quantization completed")
        return model


# =============================================================================
# QAT Quantizer
# =============================================================================


class QATQuantizer(BaseQuantizer):
    """
    Quantization-Aware Training.

    Simulates quantization during training, allowing the model to learn
    to compensate for quantization error.

    Benefits:
        - Best accuracy retention
        - Model adapts to quantization
        - Works well for aggressive quantization

    Limitations:
        - Requires retraining
        - More complex workflow
        - Takes more time
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__(config)

    def _fuse_modules(self, model: "nn.Module") -> "nn.Module":
        """Fuse modules for better quantization."""
        if self.config.fuse_modules:
            for fuse_list in self.config.fuse_modules:
                try:
                    torch.ao.quantization.fuse_modules(model, fuse_list, inplace=True)
                except Exception as e:
                    logger.warning(f"Could not fuse {fuse_list}: {e}")
        return model

    def prepare_qat(
        self,
        model: "nn.Module",
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Prepare model for QAT.

        Args:
            model: PyTorch model
            device: Device to use

        Returns:
            Model prepared for QAT
        """
        model.train()

        # Fuse modules
        model = self._fuse_modules(model)

        # Set qconfig
        qconfig = get_default_qat_qconfig(self.config.backend)
        model.qconfig = qconfig

        # Prepare for QAT
        torch.ao.quantization.prepare_qat(model, inplace=True)

        model.to(device)
        logger.info("Model prepared for QAT")
        return model

    def train_qat(
        self,
        model: "nn.Module",
        train_loader: "DataLoader",
        val_loader: Optional["DataLoader"] = None,
        criterion: Optional[Callable] = None,
        device: str = 'cpu',
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> "nn.Module":
        """
        Train with QAT.

        Args:
            model: Prepared QAT model
            train_loader: Training data
            val_loader: Validation data (optional)
            criterion: Loss function
            device: Device to use
            epochs: Number of epochs
            learning_rate: Learning rate

        Returns:
            Trained QAT model
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if epochs is None:
            epochs = self.config.qat_epochs
        if learning_rate is None:
            learning_rate = self.config.qat_learning_rate

        model.train()
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / len(train_loader)

            # Validate
            if val_loader is not None:
                val_result = self._evaluate(model, val_loader, device)
                logger.info(f"QAT Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                           f"train_acc={train_acc:.4f}, val_acc={val_result['accuracy']:.4f}")
            else:
                logger.info(f"QAT Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                           f"train_acc={train_acc:.4f}")

        return model

    def _evaluate(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        device: str
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': total_loss / len(dataloader)
        }

    def quantize(
        self,
        model: "nn.Module",
        train_loader: "DataLoader",
        val_loader: Optional["DataLoader"] = None,
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Full QAT pipeline.

        Args:
            model: PyTorch model
            train_loader: Training data
            val_loader: Validation data
            device: Device to use

        Returns:
            Quantized model after QAT
        """
        # Create copy
        model = copy.deepcopy(model)

        # Prepare for QAT
        model = self.prepare_qat(model, device)

        # Train with QAT
        model = self.train_qat(model, train_loader, val_loader, device=device)

        # Freeze quantization parameters and convert
        model.eval()
        model.to(device)
        torch.ao.quantization.convert(model, inplace=True)

        logger.info("QAT completed and model converted")
        return model


# =============================================================================
# INT4 Quantizer
# =============================================================================


class INT4Quantizer(BaseQuantizer):
    """
    INT4 quantization for extreme compression.

    Uses 4-bit weights for maximum compression ratio.
    May require QAT for good accuracy.

    Benefits:
        - 8x model size reduction (vs FP32)
        - Maximum compression

    Limitations:
        - Significant accuracy drop possible
        - Not all hardware supports INT4
        - May need QAT for good results
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__(config)

    def quantize(
        self,
        model: "nn.Module",
        use_qat: bool = True,
        train_loader: Optional["DataLoader"] = None,
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Apply INT4 quantization.

        Note: PyTorch native INT4 support is limited.
        This uses a custom approach with weight quantization.

        Args:
            model: PyTorch model
            use_qat: Whether to use QAT for better accuracy
            train_loader: Training data for QAT
            device: Device to use

        Returns:
            Quantized model
        """
        model.eval()
        model = copy.deepcopy(model)

        # For INT4, we simulate by quantizing weights manually
        # This is a simplified approach - real INT4 requires specialized kernels

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights to INT4 range [-8, 7]
                weight = module.weight.data
                w_min = weight.min()
                w_max = weight.max()

                # Scale to [-8, 7] range (4-bit signed)
                scale = (w_max - w_min) / 15.0
                if scale == 0:
                    scale = 1.0

                # Quantize
                quantized = torch.clamp(torch.round(weight / scale), -8, 7)
                # Dequantize (store as float but with INT4 values)
                module.weight.data = quantized * scale

                logger.debug(f"INT4 quantized {name}: scale={scale:.6f}")

        logger.info("INT4 weight quantization applied (simulated)")
        return model


# =============================================================================
# Quantization Manager
# =============================================================================


class QuantizationManager:
    """
    High-level quantization manager.

    Provides:
        - Automatic quantizer selection
        - Compression benchmarking
        - ONNX export
        - Model comparison utilities
    """

    QUANTIZER_CLASSES = {
        QuantizationType.DYNAMIC: DynamicQuantizer,
        QuantizationType.STATIC: StaticQuantizer,
        QuantizationType.QAT: QATQuantizer,
    }

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantization manager.

        Args:
            config: Quantization configuration
        """
        if config is None:
            config = QuantizationConfig()
        self.config = config
        self._original_model: Optional["nn.Module"] = None
        self._quantized_model: Optional["nn.Module"] = None

    def save_original_model(self, model: "nn.Module") -> None:
        """Save original model for comparison."""
        self._original_model = copy.deepcopy(model)

    def quantize(
        self,
        model: "nn.Module",
        calibration_loader: Optional["DataLoader"] = None,
        train_loader: Optional["DataLoader"] = None,
        val_loader: Optional["DataLoader"] = None,
        device: str = 'cpu'
    ) -> "nn.Module":
        """
        Apply quantization based on configuration.

        Args:
            model: PyTorch model
            calibration_loader: Data for static quantization calibration
            train_loader: Data for QAT training
            val_loader: Validation data for QAT
            device: Device to use

        Returns:
            Quantized model
        """
        # Create appropriate quantizer
        if self.config.dtype == QuantizationDtype.INT4:
            quantizer = INT4Quantizer(self.config)
            self._quantized_model = quantizer.quantize(
                model,
                use_qat=(self.config.qtype == QuantizationType.QAT),
                train_loader=train_loader,
                device=device
            )
        else:
            quantizer_class = self.QUANTIZER_CLASSES.get(self.config.qtype)
            if quantizer_class is None:
                raise ValueError(f"Unknown quantization type: {self.config.qtype}")

            quantizer = quantizer_class(self.config)

            if self.config.qtype == QuantizationType.DYNAMIC:
                self._quantized_model = quantizer.quantize(model, self.config.dtype)
            elif self.config.qtype == QuantizationType.STATIC:
                if calibration_loader is None:
                    raise ValueError("Static quantization requires calibration data")
                self._quantized_model = quantizer.quantize(model, calibration_loader, device)
            elif self.config.qtype == QuantizationType.QAT:
                if train_loader is None:
                    raise ValueError("QAT requires training data")
                self._quantized_model = quantizer.quantize(model, train_loader, val_loader, device)

        return self._quantized_model

    def get_compression_stats(
        self,
        quantized_model: Optional["nn.Module"] = None
    ) -> Dict[str, Any]:
        """
        Get compression statistics.

        Args:
            quantized_model: Quantized model (uses internal if None)

        Returns:
            Compression statistics dictionary
        """
        if quantized_model is None:
            quantized_model = self._quantized_model

        if quantized_model is None or self._original_model is None:
            raise ValueError("Need both original and quantized models")

        base = BaseQuantizer(self.config)
        original_size = base.get_model_size_mb(self._original_model)
        quantized_size = base.get_model_size_mb(quantized_model)

        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0,
            'original_params': base.count_parameters(self._original_model),
            'quantized_params': base.count_parameters(quantized_model),
            'dtype': self.config.dtype.value,
            'qtype': self.config.qtype.value
        }

    def export_onnx(
        self,
        model: "nn.Module",
        output_path: str,
        input_shape: Tuple[int, ...],
        opset_version: int = 13
    ) -> None:
        """
        Export quantized model to ONNX.

        Args:
            model: Quantized model
            output_path: Output file path
            input_shape: Input tensor shape (without batch)
            opset_version: ONNX opset version
        """
        model.eval()

        dummy_input = torch.randn(1, *input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"Model exported to {output_path}")

    def benchmark_inference(
        self,
        original_model: "nn.Module",
        quantized_model: "nn.Module",
        input_shape: Tuple[int, ...],
        device: str = 'cpu',
        n_runs: int = 100
    ) -> Dict[str, float]:
        """
        Compare inference speed.

        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            input_shape: Input tensor shape
            device: Device to use
            n_runs: Number of runs

        Returns:
            Benchmark results
        """
        base_quantizer = BaseQuantizer(self.config)

        original_time = base_quantizer.measure_inference_time(
            original_model, input_shape, device, n_runs
        )
        quantized_time = base_quantizer.measure_inference_time(
            quantized_model, input_shape, device, n_runs
        )

        return {
            'original_time_ms': original_time,
            'quantized_time_ms': quantized_time,
            'speedup': original_time / quantized_time if quantized_time > 0 else 1.0
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_quantizer(
    qtype: str = "static",
    dtype: str = "int8",
    **kwargs
) -> BaseQuantizer:
    """
    Factory function to create a quantizer.

    Args:
        qtype: Quantization type (dynamic, static, qat)
        dtype: Data type (int8, int4, fp16)
        **kwargs: Additional configuration options

    Returns:
        Quantizer instance
    """
    qtype_map = {
        'dynamic': QuantizationType.DYNAMIC,
        'static': QuantizationType.STATIC,
        'qat': QuantizationType.QAT,
    }

    dtype_map = {
        'int8': QuantizationDtype.INT8,
        'int4': QuantizationDtype.INT4,
        'fp16': QuantizationDtype.FP16,
        'bfloat16': QuantizationDtype.BF16,
    }

    if qtype not in qtype_map:
        raise ValueError(f"Unknown qtype: {qtype}. Choose from {list(qtype_map.keys())}")
    if dtype not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype}. Choose from {list(dtype_map.keys())}")

    config = QuantizationConfig(
        qtype=qtype_map[qtype],
        dtype=dtype_map[dtype],
        **kwargs
    )

    if dtype == 'int4':
        return INT4Quantizer(config)

    quantizer_class = QuantizationManager.QUANTIZER_CLASSES[config.qtype]
    return quantizer_class(config)


def quantize_model(
    model: "nn.Module",
    qtype: str = "static",
    dtype: str = "int8",
    calibration_loader: Optional["DataLoader"] = None,
    **kwargs
) -> "nn.Module":
    """
    Convenience function to quantize a model.

    Args:
        model: PyTorch model
        qtype: Quantization type
        dtype: Target data type
        calibration_loader: Calibration data for static quantization
        **kwargs: Additional options

    Returns:
        Quantized model
    """
    manager = QuantizationManager(QuantizationConfig(
        qtype=QuantizationType(qtype),
        dtype=QuantizationDtype(dtype),
        **kwargs
    ))
    return manager.quantize(model, calibration_loader=calibration_loader)


def get_quantized_model_size(model: "nn.Module") -> float:
    """
    Get quantized model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    quantizer = BaseQuantizer()
    return quantizer.get_model_size_mb(model)


# =============================================================================
# Registry
# =============================================================================


QUANTIZATION_COMPONENTS = {
    'QuantizationType': QuantizationType,
    'QuantizationDtype': QuantizationDtype,
    'ObserverType': ObserverType,
    'QuantizationConfig': QuantizationConfig,
    'BaseQuantizer': BaseQuantizer,
    'DynamicQuantizer': DynamicQuantizer,
    'StaticQuantizer': StaticQuantizer,
    'QATQuantizer': QATQuantizer,
    'INT4Quantizer': INT4Quantizer,
    'QuantizationManager': QuantizationManager,
    'create_quantizer': create_quantizer,
    'quantize_model': quantize_model,
    'get_quantized_model_size': get_quantized_model_size,
}
