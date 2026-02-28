"""
TensorRT Inference Implementation for High-Performance GPU Deployment.

This module provides:
    - TensorRTEngine: Build and optimize TensorRT engines from ONNX
    - FP16/INT8 quantization with calibration
    - Performance benchmarking vs PyTorch/ONNX Runtime
    - Memory-efficient inference on NVIDIA GPUs

Theory:
    TensorRT is NVIDIA's high-performance deep learning inference optimizer.
    It provides significant speedups through:

        1. Layer Fusion: Combines operations for fewer kernel launches
        2. Precision Calibration: FP16/INT8 quantization with minimal accuracy loss
        3. Kernel Auto-Tuning: Selects optimal CUDA kernels for target GPU
        4. Dynamic Tensor Memory: Efficient memory reuse across layers
        5. Multi-Stream Execution: Parallel inference on multiple streams

    Quantization Modes:
        - FP32: Full precision (baseline)
        - FP16: Half precision, 2x speedup, minimal accuracy loss
        - INT8: 8-bit integer, 4x speedup, requires calibration

    Calibration Process (INT8):
        1. Collect activation statistics from representative data
        2. Compute optimal scale factors using entropy minimization
        3. Generate calibration cache for future builds

Performance Expectations:
    - FP16: 2-3x faster than PyTorch inference
    - INT8: 3-4x faster than PyTorch, <1% accuracy loss with proper calibration

References:
    - TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/
    - INT8 Calibration: https://docs.nvidia.com/deeplearning/tensorrt/archives/
    - PyTorch-Quantization: https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    Tensor = None
    DataLoader = None
    Dataset = None

# Try to import TensorRT
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    trt = None

# Try to import pycuda for GPU memory operations
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    cuda = None

# Try to import onnx
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    onnx = None

logger = logging.getLogger(__name__)

# TensorRT logger singleton
TRT_LOGGER = None


def get_trt_logger(min_severity: int = trt.Logger.WARNING if HAS_TENSORRT else 3) -> Optional["trt.Logger"]:
    """Get or create TensorRT logger singleton."""
    global TRT_LOGGER
    if not HAS_TENSORRT:
        return None
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(min_severity)
    return TRT_LOGGER


# =============================================================================
# Enums and Configuration
# =============================================================================


class PrecisionMode(Enum):
    """TensorRT precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class CalibrationAlgorithm(Enum):
    """INT8 calibration algorithms."""
    ENTROPY_CALIBRATOR_2 = "entropy_calibrator_2"  # Recommended for CNN
    MINMAX_CALIBRATOR = "minmax_calibrator"  # Fast, less accurate
    LEGACY_CALIBRATOR = "legacy_calibrator"  # For backward compatibility


@dataclass
class TensorRTConfig:
    """
    Configuration for TensorRT engine building.

    Attributes:
        precision: Target precision mode (FP32, FP16, INT8)
        max_batch_size: Maximum batch size for engine
        max_workspace_size: Maximum GPU memory for tactics (in bytes)
        enable_fp16: Enable FP16 tactics
        enable_int8: Enable INT8 tactics
        enable_tactic_sources: Enable all tactic sources
        strict_type_constraints: Enforce strict type constraints
        calibration_algorithm: Algorithm for INT8 calibration
        calibration_batches: Number of batches for calibration
        calibration_cache: Path to calibration cache file
        dynamic_shapes: Dynamic shape specifications
        optimization_profiles: Optimization profiles for dynamic shapes
        dlacore: DLA core index (-1 for GPU only)
        allow_gpu_fallback: Allow GPU fallback for DLA
        timing_cache: Path to timing cache file
        use_timing_cache: Enable timing cache for faster builds
        force_timing_cache: Force using timing cache even if mismatched
    """
    precision: PrecisionMode = PrecisionMode.FP16
    max_batch_size: int = 32
    max_workspace_size: int = 1 << 30  # 1GB default
    enable_fp16: bool = True
    enable_int8: bool = False
    enable_tactic_sources: bool = True
    strict_type_constraints: bool = False
    calibration_algorithm: CalibrationAlgorithm = CalibrationAlgorithm.ENTROPY_CALIBRATOR_2
    calibration_batches: int = 100
    calibration_cache: Optional[str] = None
    dynamic_shapes: Optional[Dict[str, Tuple[int, int, int]]] = None
    optimization_profiles: Optional[List[Dict[str, Tuple]]] = None
    dlacore: int = -1
    allow_gpu_fallback: bool = True
    timing_cache: Optional[str] = None
    use_timing_cache: bool = True
    force_timing_cache: bool = False

    def __post_init__(self):
        """Validate and adjust configuration."""
        if self.precision == PrecisionMode.INT8:
            self.enable_int8 = True
        if self.precision == PrecisionMode.FP16:
            self.enable_fp16 = True


@dataclass
class InferenceResult:
    """
    Result from TensorRT inference.

    Attributes:
        outputs: Dictionary of output tensors
        inference_time_ms: Inference time in milliseconds
        preprocessing_time_ms: Preprocessing time in milliseconds
        postprocessing_time_ms: Postprocessing time in milliseconds
        total_time_ms: Total time including all stages
        gpu_memory_mb: GPU memory used for inference
    """
    outputs: Dict[str, np.ndarray]
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0
    gpu_memory_mb: float = 0.0


# =============================================================================
# INT8 Calibrator Classes
# =============================================================================


class BaseCalibrator:
    """Base class for TensorRT INT8 calibrators."""

    def __init__(
        self,
        calibration_data: Union[DataLoader, List[np.ndarray], np.ndarray],
        cache_file: str = "calibration.cache",
        input_name: str = "input"
    ):
        """
        Initialize calibrator.

        Args:
            calibration_data: Calibration dataset (DataLoader, list of arrays, or single array)
            cache_file: Path to calibration cache file
            input_name: Name of input tensor in ONNX model
        """
        self.cache_file = cache_file
        self.input_name = input_name
        self.current_index = 0
        self.batch_size = 1

        # Process calibration data
        if HAS_TORCH and isinstance(calibration_data, DataLoader):
            self.calibration_data = list(calibration_data)
            if len(self.calibration_data) > 0:
                first_batch = self.calibration_data[0]
                if isinstance(first_batch, (list, tuple)):
                    first_batch = first_batch[0]
                self.batch_size = first_batch.shape[0] if hasattr(first_batch, 'shape') else 1
        elif isinstance(calibration_data, list):
            self.calibration_data = calibration_data
            if len(calibration_data) > 0:
                self.batch_size = calibration_data[0].shape[0] if hasattr(calibration_data[0], 'shape') else 1
        elif isinstance(calibration_data, np.ndarray):
            # Single array - split into batches
            self.calibration_data = [calibration_data]
            self.batch_size = calibration_data.shape[0] if len(calibration_data.shape) > 0 else 1
        else:
            self.calibration_data = []

        # Allocate device memory for current batch
        self.device_input = None
        if HAS_PYCUDA and len(self.calibration_data) > 0:
            self._allocate_device_memory()

    def _allocate_device_memory(self):
        """Allocate GPU memory for calibration batch."""
        first_batch = self.calibration_data[0]
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        if HAS_TORCH and isinstance(first_batch, Tensor):
            first_batch = first_batch.cpu().numpy()
        self.device_input = cuda.mem_alloc(first_batch.nbytes)

    def _get_batch_array(self, index: int) -> Optional[np.ndarray]:
        """Get calibration batch as numpy array."""
        if index >= len(self.calibration_data):
            return None

        batch = self.calibration_data[index]
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if HAS_TORCH and isinstance(batch, Tensor):
            batch = batch.detach().cpu().numpy()

        # Ensure contiguous array with correct dtype
        if not batch.flags['C_CONTIGUOUS']:
            batch = np.ascontiguousarray(batch)
        return batch.astype(np.float32)


if HAS_TENSORRT:
    class EntropyCalibrator2(trt.IInt8EntropyCalibrator2, BaseCalibrator):
        """
        INT8 calibrator using entropy calibration (recommended for CNNs).

        This calibrator uses entropy minimization to find optimal quantization
        thresholds, providing good accuracy with moderate calibration time.
        """

        def __init__(
            self,
            calibration_data: Union[DataLoader, List[np.ndarray], np.ndarray],
            cache_file: str = "calibration.cache",
            input_name: str = "input"
        ):
            trt.IInt8EntropyCalibrator2.__init__(self)
            BaseCalibrator.__init__(self, calibration_data, cache_file, input_name)

        def get_batch_size(self) -> int:
            """Return batch size."""
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            """Get next calibration batch."""
            batch = self._get_batch_array(self.current_index)
            if batch is None:
                return None

            if HAS_PYCUDA and self.device_input is not None:
                cuda.memcpy_htod(self.device_input, batch)
                self.current_index += 1
                return [int(self.device_input)]
            else:
                # Fallback without PyCUDA
                self.current_index += 1
                return [batch]

        def read_calibration_cache(self) -> Optional[bytes]:
            """Read calibration cache from file."""
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            """Write calibration cache to file."""
            if self.cache_file:
                os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

    class MinMaxCalibrator(trt.IInt8MinMaxCalibrator, BaseCalibrator):
        """
        INT8 calibrator using min-max calibration.

        Faster than entropy calibration but may have slightly lower accuracy.
        Good for quick testing or when calibration time is critical.
        """

        def __init__(
            self,
            calibration_data: Union[DataLoader, List[np.ndarray], np.ndarray],
            cache_file: str = "calibration.cache",
            input_name: str = "input"
        ):
            trt.IInt8MinMaxCalibrator.__init__(self)
            BaseCalibrator.__init__(self, calibration_data, cache_file, input_name)

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            batch = self._get_batch_array(self.current_index)
            if batch is None:
                return None

            if HAS_PYCUDA and self.device_input is not None:
                cuda.memcpy_htod(self.device_input, batch)
                self.current_index += 1
                return [int(self.device_input)]
            else:
                self.current_index += 1
                return [batch]

        def read_calibration_cache(self) -> Optional[bytes]:
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            if self.cache_file:
                os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
else:
    # Placeholder classes when TensorRT is not available
    EntropyCalibrator2 = None
    MinMaxCalibrator = None


# =============================================================================
# TensorRT Engine Builder
# =============================================================================


class TensorRTEngine:
    """
    TensorRT Engine Builder and Inference.

    This class handles:
        - Building TensorRT engines from ONNX models
        - FP16/INT8 quantization with calibration
        - High-performance inference with memory optimization
        - Performance benchmarking

    Example:
        >>> # Build FP16 engine
        >>> config = TensorRTConfig(precision=PrecisionMode.FP16)
        >>> engine = TensorRTEngine(config)
        >>> engine.build_from_onnx("model.onnx", "model_fp16.engine")
        >>>
        >>> # Run inference
        >>> result = engine.inference(input_data)
        >>> print(f"Inference time: {result.inference_time_ms:.2f}ms")
    """

    def __init__(self, config: Optional[TensorRTConfig] = None):
        """
        Initialize TensorRT engine builder.

        Args:
            config: TensorRT configuration. Uses defaults if None.
        """
        self.config = config or TensorRTConfig()
        self.logger = get_trt_logger()
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = None
        self.input_names = []
        self.output_names = []
        self.binding_shapes = {}

        # Memory management
        self.device_buffers = {}
        self.host_buffers = {}

        # Check availability
        self._check_availability()

    def _check_availability(self) -> None:
        """Check TensorRT and CUDA availability."""
        if not HAS_TENSORRT:
            warnings.warn(
                "TensorRT not available. Install with: pip install tensorrt",
                RuntimeWarning
            )
        if not HAS_PYCUDA:
            warnings.warn(
                "PyCUDA not available. Install with: pip install pycuda",
                RuntimeWarning
            )

    def build_from_onnx(
        self,
        onnx_path: str,
        engine_path: Optional[str] = None,
        calibration_data: Optional[Union[DataLoader, List[np.ndarray], np.ndarray]] = None
    ) -> bool:
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model file
            engine_path: Path to save engine file (optional)
            calibration_data: Calibration data for INT8 mode

        Returns:
            True if successful, False otherwise
        """
        if not HAS_TENSORRT:
            logger.error("TensorRT not available")
            return False

        if not os.path.exists(onnx_path):
            logger.error(f"ONNX file not found: {onnx_path}")
            return False

        logger.info(f"Building TensorRT engine from: {onnx_path}")
        logger.info(f"Precision: {self.config.precision.value}")

        try:
            # Create builder
            builder = trt.Builder(self.logger)

            # Create network with explicit batch flag
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)

            # Create config
            build_config = builder.create_builder_config()

            # Set workspace size
            build_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                self.config.max_workspace_size
            )

            # Enable precision modes
            if self.config.enable_fp16:
                build_config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 mode enabled")

            if self.config.enable_int8:
                build_config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 mode enabled")

                # Setup calibrator
                if calibration_data is not None:
                    calibrator = self._create_calibrator(calibration_data)
                    build_config.int8_calibrator = calibrator
                    logger.info(f"INT8 calibrator configured with {self.config.calibration_batches} batches")

            # Enable tactic sources
            if self.config.enable_tactic_sources:
                build_config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) |
                                                1 << int(trt.TacticSource.CUBLASLT))

            # Setup timing cache
            if self.config.use_timing_cache and self.config.timing_cache:
                timing_cache = self._load_timing_cache(build_config, self.config.timing_cache)

            # Parse ONNX model
            parser = trt.OnnxParser(network, self.logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(error)}")
                    return False

            logger.info(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
            logger.info(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

            # Handle dynamic shapes
            if self.config.dynamic_shapes:
                self._setup_optimization_profiles(builder, network, build_config)

            # Build engine
            logger.info("Building engine (this may take a while)...")
            start_time = time.time()

            # Use build_serialized_network for TensorRT 8.5+
            serialized_engine = builder.build_serialized_network(network, build_config)

            if serialized_engine is None:
                logger.error("Failed to build serialized engine")
                return False

            build_time = time.time() - start_time
            logger.info(f"Engine built in {build_time:.2f} seconds")

            # Deserialize engine
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)

            if self.engine is None:
                logger.error("Failed to deserialize engine")
                return False

            # Save engine to file
            if engine_path:
                with open(engine_path, "wb") as f:
                    f.write(serialized_engine)
                logger.info(f"Engine saved to: {engine_path}")

            # Create execution context
            self.context = self.engine.create_execution_context()
            self._setup_bindings()

            return True

        except Exception as e:
            logger.error(f"Error building engine: {e}")
            return False

    def load_engine(self, engine_path: str) -> bool:
        """
        Load TensorRT engine from file.

        Args:
            engine_path: Path to engine file

        Returns:
            True if successful, False otherwise
        """
        if not HAS_TENSORRT:
            logger.error("TensorRT not available")
            return False

        if not os.path.exists(engine_path):
            logger.error(f"Engine file not found: {engine_path}")
            return False

        try:
            runtime = trt.Runtime(self.logger)
            with open(engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                logger.error("Failed to load engine")
                return False

            self.context = self.engine.create_execution_context()
            self._setup_bindings()

            logger.info(f"Engine loaded from: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading engine: {e}")
            return False

    def _create_calibrator(
        self,
        calibration_data: Union[DataLoader, List[np.ndarray], np.ndarray]
    ) -> Optional["trt.IInt8Calibrator"]:
        """Create INT8 calibrator based on configuration."""
        cache_file = self.config.calibration_cache or "calibration.cache"

        # Limit calibration batches
        if isinstance(calibration_data, list):
            calibration_data = calibration_data[:self.config.calibration_batches]
        elif isinstance(calibration_data, np.ndarray):
            calibration_data = [calibration_data]

        if self.config.calibration_algorithm == CalibrationAlgorithm.ENTROPY_CALIBRATOR_2:
            return EntropyCalibrator2(calibration_data, cache_file)
        elif self.config.calibration_algorithm == CalibrationAlgorithm.MINMAX_CALIBRATOR:
            return MinMaxCalibrator(calibration_data, cache_file)
        else:
            return EntropyCalibrator2(calibration_data, cache_file)

    def _load_timing_cache(self, config: "trt.IBuilderConfig", cache_path: str) -> Optional["trt.ITimingCache"]:
        """Load timing cache for faster builds."""
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache_data = f.read()
            timing_cache = config.create_timing_cache(cache_data)
            logger.info(f"Timing cache loaded from: {cache_path}")
        else:
            timing_cache = config.create_timing_cache(b"")
            logger.info("Created new timing cache")
        return timing_cache

    def _setup_optimization_profiles(
        self,
        builder: "trt.Builder",
        network: "trt.INetworkDefinition",
        config: "trt.IBuilderConfig"
    ) -> None:
        """Setup optimization profiles for dynamic shapes."""
        if not self.config.optimization_profiles:
            return

        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in self.config.optimization_profiles[0].items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    def _setup_bindings(self) -> None:
        """Setup input/output bindings for inference."""
        self.input_names = []
        self.output_names = []
        self.binding_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)

            self.binding_shapes[name] = (shape, dtype)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.info(f"Input tensors: {self.input_names}")
        logger.info(f"Output tensors: {self.output_names}")

    def inference(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        warmup_iterations: int = 3,
        return_timing: bool = True
    ) -> Union[InferenceResult, Dict[str, np.ndarray]]:
        """
        Run inference with TensorRT engine.

        Args:
            inputs: Input tensor(s) as numpy array or dict
            warmup_iterations: Number of warmup iterations
            return_timing: Whether to return timing information

        Returns:
            InferenceResult with outputs and timing, or just outputs dict
        """
        if self.engine is None or self.context is None:
            raise RuntimeError("Engine not loaded. Call build_from_onnx() or load_engine() first.")

        total_start = time.time()

        # Prepare inputs
        prep_start = time.time()
        if isinstance(inputs, np.ndarray):
            if len(self.input_names) == 1:
                inputs = {self.input_names[0]: inputs}
            else:
                raise ValueError(f"Model expects {len(self.input_names)} inputs, got single array")

        # Ensure correct dtype and contiguity
        for name, arr in inputs.items():
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            inputs[name] = arr.astype(np.float32)
        prep_time = (time.time() - prep_start) * 1000

        # Warmup
        for _ in range(warmup_iterations):
            self._run_inference(inputs)

        # Timed inference
        inf_start = time.time()
        outputs = self._run_inference(inputs)
        inf_time = (time.time() - inf_start) * 1000

        total_time = (time.time() - total_start) * 1000

        if return_timing:
            return InferenceResult(
                outputs=outputs,
                inference_time_ms=inf_time,
                preprocessing_time_ms=prep_time,
                total_time_ms=total_time
            )
        return outputs

    def _run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute single inference."""
        if HAS_PYCUDA:
            return self._run_inference_cuda(inputs)
        else:
            return self._run_inference_numpy(inputs)

    def _run_inference_cuda(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference using CUDA (PyCUDA)."""
        # Create stream
        if self.stream is None:
            self.stream = cuda.Stream()

        # Allocate and copy input buffers
        device_buffers = {}
        for name, arr in inputs.items():
            # Set input shape for dynamic shapes
            self.context.set_input_shape(name, arr.shape)

            device_buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(device_buf, arr, self.stream)
            device_buffers[name] = device_buf

        # Allocate output buffers
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            output_arr = np.empty(shape, dtype=dtype)
            device_buf = cuda.mem_alloc(output_arr.nbytes)
            device_buffers[name] = device_buf
            outputs[name] = output_arr

        # Set tensor addresses
        for name, buf in device_buffers.items():
            self.context.set_tensor_address(name, int(buf))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        for name in self.output_names:
            cuda.memcpy_dtoh_async(outputs[name], device_buffers[name], self.stream)

        self.stream.synchronize()

        return outputs

    def _run_inference_numpy(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference using numpy (CPU fallback, slower)."""
        # Set input shapes
        for name, arr in inputs.items():
            self.context.set_input_shape(name, arr.shape)

        # Prepare output buffers
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            outputs[name] = np.empty(shape, dtype=dtype)

        # Set tensor addresses (using numpy array data pointers)
        for name, arr in inputs.items():
            self.context.set_tensor_address(name, arr.ctypes.data)

        for name, arr in outputs.items():
            self.context.set_tensor_address(name, arr.ctypes.data)

        # Execute
        self.context.execute_async_v3(stream_handle=0)

        return outputs

    def benchmark(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            inputs: Input tensor(s)
            iterations: Number of benchmark iterations
            warmup_iterations: Warmup iterations

        Returns:
            Dictionary with benchmark results
        """
        if isinstance(inputs, np.ndarray):
            inputs = {self.input_names[0]: inputs}

        # Warmup
        for _ in range(warmup_iterations):
            self._run_inference(inputs)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self._run_inference(inputs)
            times.append((time.time() - start) * 1000)

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "iterations": iterations,
        }

    def get_memory_usage(self) -> Dict[str, int]:
        """Get GPU memory usage statistics."""
        if not HAS_PYCUDA:
            return {}

        try:
            import pycuda.autoinit
            free, total = cuda.mem_get_info()
            return {
                "gpu_memory_free_mb": free / (1024 * 1024),
                "gpu_memory_total_mb": total / (1024 * 1024),
                "gpu_memory_used_mb": (total - free) / (1024 * 1024),
            }
        except Exception:
            return {}


# =============================================================================
# Benchmarking Utilities
# =============================================================================


class TensorRTBenchmark:
    """
    Compare TensorRT vs PyTorch/ONNX Runtime performance.

    Example:
        >>> benchmark = TensorRTBenchmark()
        >>> results = benchmark.compare_inference(
        ...     pytorch_model=model,
        ...     onnx_path="model.onnx",
        ...     input_shape=(1, 3, 224, 224),
        ...     iterations=100
        ... )
        >>> print(f"TensorRT speedup: {results['speedup']:.2f}x")
    """

    def __init__(self, tensorrt_config: Optional[TensorRTConfig] = None):
        """
        Initialize benchmark.

        Args:
            tensorrt_config: TensorRT configuration
        """
        self.tensorrt_config = tensorrt_config or TensorRTConfig()

    def compare_inference(
        self,
        pytorch_model: Optional["nn.Module"] = None,
        onnx_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        input_dtype: np.dtype = np.float32,
        iterations: int = 100,
        warmup_iterations: int = 10,
        calibration_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare inference performance across frameworks.

        Args:
            pytorch_model: PyTorch model (optional)
            onnx_path: Path to ONNX model
            input_shape: Shape of input tensor
            input_dtype: Input data type
            iterations: Number of benchmark iterations
            warmup_iterations: Warmup iterations
            calibration_data: Calibration data for INT8

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        # Generate random input
        input_data = np.random.randn(*input_shape).astype(input_dtype)

        # Benchmark PyTorch
        if pytorch_model is not None and HAS_TORCH:
            results["pytorch"] = self._benchmark_pytorch(
                pytorch_model, input_data, iterations, warmup_iterations
            )

        # Benchmark TensorRT
        if onnx_path is not None and HAS_TENSORRT:
            engine = TensorRTEngine(self.tensorrt_config)

            # Build engine
            engine_path = onnx_path.replace(".onnx", f"_{self.tensorrt_config.precision.value}.engine")
            if os.path.exists(engine_path):
                engine.load_engine(engine_path)
            else:
                engine.build_from_onnx(onnx_path, engine_path, calibration_data)

            results["tensorrt"] = engine.benchmark(input_data, iterations, warmup_iterations)
            results["tensorrt"]["memory_mb"] = engine.get_memory_usage().get("gpu_memory_used_mb", 0)

        # Calculate speedup
        if "pytorch" in results and "tensorrt" in results:
            results["speedup"] = results["pytorch"]["mean_ms"] / results["tensorrt"]["mean_ms"]

        return results

    def _benchmark_pytorch(
        self,
        model: "nn.Module",
        input_data: np.ndarray,
        iterations: int,
        warmup_iterations: int
    ) -> Dict[str, float]:
        """Benchmark PyTorch model."""
        model.eval()
        device = next(model.parameters()).device
        input_tensor = torch.from_numpy(input_data).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = model(input_tensor)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                else:
                    import time
                    start = time.time()
                    _ = model(input_tensor)
                    times.append((time.time() - start) * 1000)

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }

    def compare_accuracy(
        self,
        pytorch_model: "nn.Module",
        tensorrt_engine: TensorRTEngine,
        test_data: np.ndarray,
        rtol: float = 1e-2,
        atol: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Compare accuracy between PyTorch and TensorRT outputs.

        Args:
            pytorch_model: PyTorch model
            tensorrt_engine: TensorRT engine
            test_data: Test input data
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Dictionary with accuracy comparison results
        """
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(torch.from_numpy(test_data))
            if isinstance(pytorch_output, (list, tuple)):
                pytorch_output = pytorch_output[0]
            pytorch_output = pytorch_output.cpu().numpy()

        # TensorRT inference
        trt_result = tensorrt_engine.inference(test_data)
        trt_output = list(trt_result.outputs.values())[0]

        # Compare
        max_diff = np.max(np.abs(pytorch_output - trt_output))
        mean_diff = np.mean(np.abs(pytorch_output - trt_output))
        is_close = np.allclose(pytorch_output, trt_output, rtol=rtol, atol=atol)

        return {
            "max_absolute_diff": float(max_diff),
            "mean_absolute_diff": float(mean_diff),
            "is_close": is_close,
            "rtol": rtol,
            "atol": atol,
            "passed": is_close,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    max_workspace_size: int = 1 << 30,
    calibration_data: Optional[np.ndarray] = None
) -> bool:
    """
    Convenience function to build TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save engine
        precision: "fp32", "fp16", or "int8"
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace in bytes
        calibration_data: Calibration data for INT8

    Returns:
        True if successful
    """
    precision_map = {
        "fp32": PrecisionMode.FP32,
        "fp16": PrecisionMode.FP16,
        "int8": PrecisionMode.INT8,
    }

    config = TensorRTConfig(
        precision=precision_map.get(precision.lower(), PrecisionMode.FP16),
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size
    )

    engine = TensorRTEngine(config)
    return engine.build_from_onnx(onnx_path, engine_path, calibration_data)


def tensorrt_inference(
    engine_path: str,
    input_data: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Convenience function for TensorRT inference.

    Args:
        engine_path: Path to TensorRT engine
        input_data: Input numpy array

    Returns:
        Dictionary of output tensors
    """
    engine = TensorRTEngine()
    engine.load_engine(engine_path)
    result = engine.inference(input_data)
    return result.outputs


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PrecisionMode",
    "CalibrationAlgorithm",
    "TensorRTConfig",
    "InferenceResult",
    "EntropyCalibrator2",
    "MinMaxCalibrator",
    "TensorRTEngine",
    "TensorRTBenchmark",
    "build_tensorrt_engine",
    "tensorrt_inference",
    "HAS_TENSORRT",
    "HAS_PYCUDA",
    "TENSORRT_COMPONENTS",
]


# Registry of all TensorRT components
TENSORRT_COMPONENTS = {
    "enums": ["PrecisionMode", "CalibrationAlgorithm"],
    "config": ["TensorRTConfig"],
    "results": ["InferenceResult"],
    "calibrators": ["EntropyCalibrator2", "MinMaxCalibrator"],
    "engine": ["TensorRTEngine"],
    "benchmark": ["TensorRTBenchmark"],
    "utilities": ["build_tensorrt_engine", "tensorrt_inference"],
    "availability": ["HAS_TENSORRT", "HAS_PYCUDA"],
}
