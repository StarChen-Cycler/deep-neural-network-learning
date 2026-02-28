"""
ONNX Model Export Implementation for PyTorch Models.

This module provides:
    - ONNXExporter: Export PyTorch models to ONNX format
    - Dynamic axes support for variable batch/sequence lengths
    - Operator validation and conversion verification
    - Model optimization and simplification

Theory:
    ONNX (Open Neural Network Exchange) is an open format for representing
    machine learning models. It enables models trained in PyTorch to be
    deployed across different frameworks and hardware platforms.

    Key Concepts:
        - Opset Version: Version of ONNX operator specifications
        - Dynamic Axes: Variable dimensions (batch size, sequence length)
        - Graph Optimization: Constant folding, dead code elimination
        - Operator Fusion: Combining operations for efficiency

    Export Process:
        1. Prepare model in eval mode
        2. Create example inputs with correct shapes
        3. Define dynamic axes for variable dimensions
        4. Export using torch.onnx.export
        5. Validate with onnx.checker
        6. Optimize with onnx.optimizer (optional)

References:
    - PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
    - ONNX Specification: https://onnx.ai/get-started.html
    - ONNX Runtime: https://onnxruntime.ai/docs/
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import logging
import tempfile
import warnings

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    Tensor = None

# Try to import onnx
try:
    import onnx
    import onnx.checker
    import onnx.optimizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    onnx = None

# Try to import onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    ort = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class OpsetVersion(Enum):
    """Supported ONNX opset versions."""
    V11 = 11  # Stable, widely supported
    V12 = 12
    V13 = 13
    V14 = 14
    V15 = 15
    V17 = 17  # Recommended for modern models
    V18 = 18
    V19 = 19
    V20 = 20  # Latest stable


class ExportMode(Enum):
    """Export modes for PyTorch models."""
    TRACING = "tracing"  # Trace execution with example inputs
    SCRIPTING = "scripting"  # Script the model (handles control flow)
    DYNAMO = "dynamo"  # PyTorch 2.0+ Dynamo export


@dataclass
class DynamicAxis:
    """Configuration for a dynamic axis."""
    name: str  # Name of the dynamic dimension (e.g., "batch_size")
    dim: int  # Dimension index (e.g., 0 for batch dimension)

    def to_dict(self) -> Dict[str, Any]:
        return {self.dim: self.name}


@dataclass
class ONNXExportConfig:
    """
    Configuration for ONNX export.

    Attributes:
        opset_version: ONNX operator set version
        export_mode: Export mode (tracing, scripting, dynamo)
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        dynamic_batch: Enable dynamic batch size (dim 0)
        dynamic_sequence: Enable dynamic sequence length (dim 1)
        optimize_model: Apply ONNX optimizations after export
        strip_doc_string: Remove documentation strings
        check_model: Validate exported model with onnx.checker
        external_data: Save large tensors as external data
        keep_initializers_as_inputs: Keep initializers as graph inputs
    """
    opset_version: OpsetVersion = OpsetVersion.V17
    export_mode: ExportMode = ExportMode.TRACING
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    dynamic_batch: bool = True
    dynamic_sequence: bool = False
    optimize_model: bool = True
    strip_doc_string: bool = False
    check_model: bool = True
    external_data: bool = False
    keep_initializers_as_inputs: bool = False

    def get_opset_version(self) -> int:
        """Get integer opset version."""
        return self.opset_version.value

    def build_dynamic_axes(
        self,
        input_names: List[str],
        output_names: List[str],
        num_sequence_inputs: int = 0
    ) -> Dict[str, Dict[int, str]]:
        """
        Build dynamic axes configuration.

        Args:
            input_names: List of input tensor names
            output_names: List of output tensor names
            num_sequence_inputs: Number of inputs with sequence dimension

        Returns:
            Dictionary mapping tensor names to dynamic axis specs
        """
        if self.dynamic_axes is not None:
            return self.dynamic_axes

        dynamic_axes = {}

        # Add dynamic batch dimension for inputs
        for i, name in enumerate(input_names):
            axes = {}
            if self.dynamic_batch:
                axes[0] = "batch_size"
            if self.dynamic_sequence and i < num_sequence_inputs:
                axes[1] = "sequence_length"
            if axes:
                dynamic_axes[name] = axes

        # Add dynamic batch dimension for outputs
        for name in output_names:
            axes = {}
            if self.dynamic_batch:
                axes[0] = "batch_size"
            if self.dynamic_sequence:
                axes[1] = "sequence_length"
            if axes:
                dynamic_axes[name] = axes

        return dynamic_axes


# =============================================================================
# ONNX Exporter
# =============================================================================


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.

    This class provides comprehensive ONNX export functionality including:
    - Dynamic axes support for variable dimensions
    - Multiple export modes (tracing, scripting, dynamo)
    - Model validation and optimization
    - Operator compatibility checking

    Example:
        >>> import torchvision.models as models
        >>> model = models.resnet50(pretrained=True)
        >>> exporter = ONNXExporter()
        >>> exporter.export(model, "resnet50.onnx", (1, 3, 224, 224))
    """

    def __init__(self, config: Optional[ONNXExportConfig] = None):
        """
        Initialize ONNX exporter.

        Args:
            config: Export configuration (uses defaults if None)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ONNX export")

        self.config = config or ONNXExportConfig()
        self._export_time: Optional[float] = None
        self._model_size: Optional[int] = None

    def export(
        self,
        model: nn.Module,
        output_path: str,
        example_inputs: Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        num_sequence_inputs: int = 0,
        **kwargs
    ) -> str:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            example_inputs: Example inputs for tracing
            input_names: Names for input tensors
            output_names: Names for output tensors
            num_sequence_inputs: Number of inputs with sequence dimension
            **kwargs: Additional export arguments

        Returns:
            Path to exported ONNX model
        """
        # Set model to eval mode
        model.eval()

        # Prepare input/output names
        if input_names is None:
            if isinstance(example_inputs, dict):
                input_names = list(example_inputs.keys())
            else:
                input_names = [f"input_{i}" for i in range(self._count_inputs(example_inputs))]

        if output_names is None:
            output_names = ["output"]

        # Build dynamic axes
        dynamic_axes = self.config.build_dynamic_axes(
            input_names, output_names, num_sequence_inputs
        )

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Export based on mode
        start_time = time.time()

        if self.config.export_mode == ExportMode.DYNAMO:
            onnx_program = self._export_dynamo(
                model, example_inputs, output_path, input_names,
                output_names, dynamic_axes, **kwargs
            )
        elif self.config.export_mode == ExportMode.SCRIPTING:
            self._export_scripting(
                model, example_inputs, output_path, input_names,
                output_names, dynamic_axes, **kwargs
            )
        else:  # TRACING (default)
            self._export_tracing(
                model, example_inputs, output_path, input_names,
                output_names, dynamic_axes, **kwargs
            )

        self._export_time = time.time() - start_time

        # Validate model
        if self.config.check_model and HAS_ONNX:
            self._validate_model(output_path)

        # Optimize model
        if self.config.optimize_model and HAS_ONNX:
            self._optimize_model(output_path)

        # Get model size
        self._model_size = os.path.getsize(output_path)

        logger.info(f"Exported ONNX model to {output_path} "
                   f"(size: {self._model_size / 1024 / 1024:.2f} MB, "
                   f"time: {self._export_time:.2f}s)")

        return output_path

    def _count_inputs(self, inputs: Union[Tensor, Tuple, Dict]) -> int:
        """Count number of input tensors."""
        if isinstance(inputs, Tensor):
            return 1
        elif isinstance(inputs, dict):
            return len(inputs)
        elif isinstance(inputs, (tuple, list)):
            return len(inputs)
        return 1

    def _export_tracing(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple, Dict],
        output_path: str,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        **kwargs
    ) -> None:
        """Export using tracing mode."""
        # Convert dict inputs to tuple for tracing
        if isinstance(example_inputs, dict):
            example_inputs = tuple(example_inputs.values())

        torch.onnx.export(
            model,
            example_inputs,
            output_path,
            opset_version=self.config.get_opset_version(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=self.config.keep_initializers_as_inputs,
            dynamo=False,  # Use tracing mode, not Dynamo
            **kwargs
        )

    def _export_scripting(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple, Dict],
        output_path: str,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        **kwargs
    ) -> None:
        """Export using scripting mode."""
        # Script the model first
        scripted_model = torch.jit.script(model)

        # Convert dict inputs to tuple
        if isinstance(example_inputs, dict):
            example_inputs = tuple(example_inputs.values())

        torch.onnx.export(
            scripted_model,
            example_inputs,
            output_path,
            opset_version=self.config.get_opset_version(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            export_params=True,
            do_constant_folding=True,
            dynamo=False,  # Use scripting mode, not Dynamo
            **kwargs
        )

    def _export_dynamo(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple, Dict],
        output_path: str,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        **kwargs
    ) -> Any:
        """Export using PyTorch 2.0+ Dynamo mode."""
        # Convert dict inputs to tuple
        if isinstance(example_inputs, dict):
            example_inputs = tuple(example_inputs.values())

        # Build dynamic shapes for dynamo
        dynamic_shapes = None
        if dynamic_axes:
            dynamic_shapes = self._build_dynamic_shapes(example_inputs, dynamic_axes, input_names)

        onnx_program = torch.onnx.export(
            model,
            example_inputs if isinstance(example_inputs, tuple) else (example_inputs,),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            opset_version=self.config.get_opset_version(),
            **kwargs
        )

        return onnx_program

    def _build_dynamic_shapes(
        self,
        example_inputs: Union[Tensor, Tuple],
        dynamic_axes: Dict[str, Dict[int, str]],
        input_names: List[str]
    ) -> Optional[Tuple]:
        """Build dynamic shapes for Dynamo export."""
        if not dynamic_axes:
            return None

        shapes = []
        inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)

        for i, (name, inp) in enumerate(zip(input_names, inputs)):
            if name in dynamic_axes and isinstance(inp, Tensor):
                # Create dynamic dimension spec
                shape_spec = {}
                for dim, dim_name in dynamic_axes[name].items():
                    shape_spec[dim] = torch.export.Dim(dim_name)
                shapes.append(shape_spec)
            else:
                shapes.append(None)

        return tuple(shapes) if shapes else None

    def _validate_model(self, model_path: str) -> bool:
        """Validate ONNX model format."""
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
            return True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise

    def _optimize_model(self, model_path: str) -> None:
        """Apply ONNX optimizations."""
        try:
            onnx_model = onnx.load(model_path)

            # Get available passes
            passes = onnx.optimizer.get_available_passes()

            # Apply common optimizations
            optimized_model = onnx.optimizer.optimize(
                onnx_model,
                passes=['eliminate_deadend', 'eliminate_identity',
                       'eliminate_nop_transpose', 'fuse_bn_into_conv',
                       'fuse_consecutive_transposes']
            )

            onnx.save(optimized_model, model_path)
            logger.info("ONNX model optimization applied")
        except Exception as e:
            logger.warning(f"ONNX optimization failed (non-critical): {e}")

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about an ONNX model.

        Args:
            model_path: Path to ONNX model

        Returns:
            Dictionary with model information
        """
        try:
            import onnx
            import onnx.checker
        except ImportError:
            raise ImportError("ONNX is required for model inspection")

        onnx_model = onnx.load(model_path)
        graph = onnx_model.graph

        info = {
            "opset_version": onnx_model.opset_import[0].version if onnx_model.opset_import else None,
            "producer": getattr(graph, "producer_name", "unknown"),
            "inputs": [],
            "outputs": [],
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
        }

        # Extract input info
        for inp in graph.input:
            if inp.name not in [init.name for init in graph.initializer]:
                shape = [d.dim_value if d.dim_value else d.dim_param
                        for d in inp.type.tensor_type.shape.dim]
                info["inputs"].append({
                    "name": inp.name,
                    "shape": shape,
                    "type": inp.type.tensor_type.elem_type
                })

        # Extract output info
        for out in graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param
                    for d in out.type.tensor_type.shape.dim]
            info["outputs"].append({
                "name": out.name,
                "shape": shape,
                "type": out.type.tensor_type.elem_type
            })

        return info

    def check_operator_support(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple],
        opset_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if all operators in the model are supported by ONNX.

        Args:
            model: PyTorch model to check
            example_inputs: Example inputs for tracing
            opset_version: ONNX opset version to check against

        Returns:
            Dictionary with support status and any issues
        """
        if opset_version is None:
            opset_version = self.config.get_opset_version()

        result = {
            "supported": True,
            "unsupported_ops": [],
            "warnings": []
        }

        try:
            # Try export to string to check for issues
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                temp_path = f.name

            # Convert dict inputs to tuple
            if isinstance(example_inputs, dict):
                example_inputs = tuple(example_inputs.values())

            torch.onnx.export(
                model,
                example_inputs,
                temp_path,
                opset_version=opset_version,
                export_params=False,  # Don't save params for checking
                dynamo=False,
            )

            os.unlink(temp_path)

        except RuntimeError as e:
            # Catch runtime errors including unsupported operators
            error_msg = str(e)
            if "Unsupported" in error_msg or "not supported" in error_msg.lower():
                result["supported"] = False
                result["unsupported_ops"].append(error_msg)
            else:
                result["warnings"].append(error_msg)
        except Exception as e:
            result["warnings"].append(str(e))

        return result

    @property
    def last_export_time(self) -> Optional[float]:
        """Get time taken for last export."""
        return self._export_time

    @property
    def last_model_size(self) -> Optional[int]:
        """Get size of last exported model in bytes."""
        return self._model_size


# =============================================================================
# Convenience Functions
# =============================================================================


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    example_inputs: Union[Tensor, Tuple, Dict],
    dynamic_batch: bool = True,
    opset_version: int = 17,
    **kwargs
) -> str:
    """
    Convenience function to export a model to ONNX.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        example_inputs: Example inputs for tracing
        dynamic_batch: Enable dynamic batch size
        opset_version: ONNX opset version
        **kwargs: Additional export arguments

    Returns:
        Path to exported model
    """
    config = ONNXExportConfig(
        opset_version=OpsetVersion(opset_version),
        dynamic_batch=dynamic_batch,
    )
    exporter = ONNXExporter(config)
    return exporter.export(model, output_path, example_inputs, **kwargs)


def export_resnet_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
) -> str:
    """
    Export ResNet-style model to ONNX with standard configuration.

    Args:
        model: ResNet model
        output_path: Path to save ONNX model
        input_size: Input size (channels, height, width)
        batch_size: Example batch size

    Returns:
        Path to exported model
    """
    example_input = torch.randn(batch_size, *input_size)
    config = ONNXExportConfig(
        opset_version=OpsetVersion.V17,
        dynamic_batch=True,
        input_names=["input"],
        output_names=["output"],
    )
    exporter = ONNXExporter(config)
    return exporter.export(
        model,
        output_path,
        example_input,
        input_names=["input"],
        output_names=["output"]
    )


def export_transformer_to_onnx(
    model: nn.Module,
    output_path: str,
    hidden_size: int,
    num_attention_heads: int,
    sequence_length: int = 128,
    batch_size: int = 1,
) -> str:
    """
    Export Transformer model to ONNX with dynamic sequence length.

    Args:
        model: Transformer model
        output_path: Path to save ONNX model
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        sequence_length: Example sequence length
        batch_size: Example batch size

    Returns:
        Path to exported model
    """
    config = ONNXExportConfig(
        opset_version=OpsetVersion.V17,
        dynamic_batch=True,
        dynamic_sequence=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
    )
    exporter = ONNXExporter(config)

    # Create example inputs for transformer
    example_inputs = {
        "input_ids": torch.randint(0, 30000, (batch_size, sequence_length)),
        "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
    }

    return exporter.export(
        model,
        output_path,
        example_inputs,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        num_sequence_inputs=2
    )


# =============================================================================
# Registry
# =============================================================================


ONNX_EXPORT_COMPONENTS = {
    "enums": ["OpsetVersion", "ExportMode"],
    "config": ["DynamicAxis", "ONNXExportConfig"],
    "exporter": ["ONNXExporter"],
    "functions": [
        "export_to_onnx",
        "export_resnet_to_onnx",
        "export_transformer_to_onnx",
    ],
}
