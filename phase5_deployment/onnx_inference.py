"""
ONNX Runtime Inference Implementation.

This module provides:
    - ONNXInference: High-performance inference with ONNX Runtime
    - Performance benchmarking vs PyTorch
    - Multi-provider support (CPU, CUDA, TensorRT)
    - Batch inference utilities

Theory:
    ONNX Runtime is a high-performance inference engine for ONNX models.
    It provides significant speedups over PyTorch inference through:

        1. Graph Optimization: Constant folding, operator fusion
        2. Hardware Acceleration: CUDA, TensorRT, OpenVINO, etc.
        3. Memory Optimization: Efficient memory allocation and reuse
        4. Quantization Support: INT8, FP16 inference

    Performance Tips:
        - Use CUDAExecutionProvider for GPU inference
        - Enable graph optimization (default)
        - Set intra_op_num_threads for CPU parallelism
        - Use IOBinding for large tensors to avoid copies

References:
    - ONNX Runtime: https://onnxruntime.ai/docs/
    - Performance Tuning: https://onnxruntime.ai/docs/performance/
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import warnings
from contextlib import contextmanager

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

# Try to import onnxruntime
try:
    import onnxruntime as ort
    from onnxruntime import GraphOptimizationLevel, SessionOptions
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    ort = None
    GraphOptimizationLevel = None
    SessionOptions = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class ExecutionProvider(Enum):
    """ONNX Runtime execution providers."""
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    COREML = "CoreMLExecutionProvider"


class GraphOptimization(Enum):
    """Graph optimization levels."""
    DISABLED = "disabled"
    BASIC = "basic"
    ALL = "all"


@dataclass
class InferenceConfig:
    """
    Configuration for ONNX Runtime inference.

    Attributes:
        execution_provider: Primary execution provider
        fallback_provider: Fallback provider if primary unavailable
        graph_optimization: Graph optimization level
        intra_op_num_threads: Threads for intra-op parallelism
        inter_op_num_threads: Threads for inter-op parallelism
        enable_memory_pattern: Enable memory pattern optimization
        enable_cpu_mem_arena: Enable CPU memory arena
        gpu_device_id: GPU device ID for CUDA provider
        enable_profiling: Enable profiling
    """
    execution_provider: ExecutionProvider = ExecutionProvider.CPU
    fallback_provider: Optional[ExecutionProvider] = None
    graph_optimization: GraphOptimization = GraphOptimization.ALL
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 1
    enable_memory_pattern: bool = True
    enable_cpu_mem_arena: bool = True
    gpu_device_id: int = 0
    enable_profiling: bool = False

    def get_providers(self) -> List[str]:
        """Get list of execution providers in priority order."""
        providers = [self.execution_provider.value]
        if self.fallback_provider:
            providers.append(self.fallback_provider.value)
        # Always add CPU as final fallback
        if ExecutionProvider.CPU.value not in providers:
            providers.append(ExecutionProvider.CPU.value)
        return providers

    def get_provider_options(self) -> List[Dict[str, Any]]:
        """Get provider-specific options."""
        options = []

        for provider in self.get_providers():
            if provider == ExecutionProvider.CUDA.value:
                options.append({
                    "device_id": self.gpu_device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                })
            elif provider == ExecutionProvider.TENSORRT.value:
                options.append({
                    "device_id": self.gpu_device_id,
                    "trt_fp16_enable": True,
                    "trt_int8_enable": False,
                })
            else:
                options.append({})

        return options

    def get_session_options(self) -> "SessionOptions":
        """Create ONNX Runtime session options."""
        if not HAS_ONNXRUNTIME:
            raise ImportError("ONNX Runtime is required")

        so = SessionOptions()

        # Graph optimization level
        if self.graph_optimization == GraphOptimization.DISABLED:
            so.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        elif self.graph_optimization == GraphOptimization.BASIC:
            so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Threading
        so.intra_op_num_threads = self.intra_op_num_threads
        so.inter_op_num_threads = self.inter_op_num_threads

        # Memory
        so.enable_mem_pattern = self.enable_memory_pattern
        so.enable_cpu_mem_arena = self.enable_cpu_mem_arena

        # Profiling
        if self.enable_profiling:
            so.enable_profiling = True

        return so


# =============================================================================
# ONNX Inference
# =============================================================================


class ONNXInference:
    """
    High-performance ONNX model inference.

    This class provides efficient inference with ONNX Runtime including:
    - Multiple execution providers (CPU, CUDA, TensorRT)
    - Automatic input/output handling
    - Batch inference utilities
    - Performance benchmarking

    Example:
        >>> inferencer = ONNXInference("model.onnx")
        >>> outputs = inferencer.run(input_array)
        >>> outputs = inferencer.run_batch([input1, input2, input3])
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize ONNX inference.

        Args:
            model_path: Path to ONNX model
            config: Inference configuration (uses defaults if None)
        """
        if not HAS_ONNXRUNTIME:
            raise ImportError("ONNX Runtime is required for inference")

        self.model_path = model_path
        self.config = config or InferenceConfig()

        # Create session
        self._session = self._create_session()

        # Get input/output metadata
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._input_shapes = [inp.shape for inp in self._session.get_inputs()]
        self._output_shapes = [out.shape for out in self._session.get_outputs()]

        logger.info(f"Loaded ONNX model from {model_path}")
        logger.info(f"Inputs: {self._input_names}, Outputs: {self._output_names}")
        logger.info(f"Execution provider: {self._session.get_providers()}")

    def _create_session(self) -> "ort.InferenceSession":
        """Create ONNX Runtime inference session."""
        so = self.config.get_session_options()
        providers = self.config.get_providers()
        provider_options = self.config.get_provider_options()

        # Filter to only available providers
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]

        if not providers:
            raise RuntimeError("No compatible execution providers available")

        session = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=providers,
            provider_options=provider_options,
        )

        return session

    @property
    def input_names(self) -> List[str]:
        """Get input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Get output tensor names."""
        return self._output_names

    @property
    def input_shapes(self) -> List[List[Union[int, str]]]:
        """Get input tensor shapes."""
        return self._input_shapes

    @property
    def output_shapes(self) -> List[List[Union[int, str]]]:
        """Get output tensor shapes."""
        return self._output_shapes

    def run(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
        output_names: Optional[List[str]] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference on a single input.

        Args:
            inputs: Input data as numpy array or dict/list
            output_names: Specific outputs to return (None = all)

        Returns:
            Output array or list of output arrays
        """
        # Prepare input feed
        input_feed = self._prepare_input_feed(inputs)

        # Get output names
        if output_names is None:
            output_names = self._output_names

        # Run inference
        outputs = self._session.run(output_names, input_feed)

        # Return single output directly
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def run_batch(
        self,
        batch_inputs: List[Union[np.ndarray, Dict[str, np.ndarray]]],
        output_names: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> List[Union[np.ndarray, List[np.ndarray]]]:
        """
        Run inference on a batch of inputs.

        Args:
            batch_inputs: List of inputs
            output_names: Specific outputs to return
            show_progress: Show progress bar

        Returns:
            List of outputs
        """
        results = []

        iterator = batch_inputs
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(batch_inputs, desc="Inference")
            except ImportError:
                pass

        for inputs in iterator:
            output = self.run(inputs, output_names)
            results.append(output)

        return results

    def _prepare_input_feed(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Prepare input feed dictionary."""
        if isinstance(inputs, np.ndarray):
            if len(self._input_names) == 1:
                return {self._input_names[0]: inputs}
            else:
                raise ValueError(
                    f"Model expects {len(self._input_names)} inputs, "
                    f"but only one array was provided"
                )
        elif isinstance(inputs, dict):
            return {name: self._ensure_numpy(inputs[name])
                   for name in self._input_names if name in inputs}
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) != len(self._input_names):
                raise ValueError(
                    f"Model expects {len(self._input_names)} inputs, "
                    f"but {len(inputs)} were provided"
                )
            return {name: self._ensure_numpy(arr)
                   for name, arr in zip(self._input_names, inputs)}
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

    def _ensure_numpy(self, arr: Any) -> np.ndarray:
        """Convert to numpy array if needed."""
        if HAS_TORCH and isinstance(arr, Tensor):
            return arr.detach().cpu().numpy()
        elif isinstance(arr, np.ndarray):
            return arr
        else:
            return np.array(arr)

    def benchmark(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            inputs: Example inputs
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs

        Returns:
            Dictionary with performance metrics
        """
        input_feed = self._prepare_input_feed(inputs)

        # Warmup
        for _ in range(warmup_runs):
            self._session.run(self._output_names, input_feed)

        # Benchmark
        latencies = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            self._session.run(self._output_names, input_feed)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput": 1000.0 / np.mean(latencies),  # inferences/sec
        }

    def get_profiling_results(self) -> Optional[Dict[str, Any]]:
        """Get profiling results if profiling was enabled."""
        if not self.config.enable_profiling:
            return None

        try:
            profile_file = self._session.end_profiling()
            return {"profile_file": profile_file}
        except Exception:
            return None

    @contextmanager
    def profiling(self, output_path: Optional[str] = None):
        """Context manager for profiling a block of code."""
        # Enable profiling
        self._session.session_options.enable_profiling = True
        if output_path:
            self._session.session_options.profile_file_prefix = output_path

        try:
            yield self
        finally:
            # Get results
            result = self.get_profiling_results()
            if result:
                logger.info(f"Profiling results: {result}")
            # Disable profiling
            self._session.session_options.enable_profiling = False


# =============================================================================
# Benchmarking Utilities
# =============================================================================


class InferenceBenchmark:
    """
    Compare PyTorch vs ONNX Runtime inference performance.

    Example:
        >>> benchmark = InferenceBenchmark(pytorch_model, "model.onnx")
        >>> results = benchmark.run(example_input)
        >>> print(f"ONNX is {results['speedup']:.2f}x faster")
    """

    def __init__(
        self,
        pytorch_model: "nn.Module",
        onnx_model_path: str,
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize benchmark.

        Args:
            pytorch_model: PyTorch model
            onnx_model_path: Path to ONNX model
            inference_config: ONNX inference configuration
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for benchmarking")

        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()

        self.onnx_inference = ONNXInference(
            onnx_model_path,
            config=inference_config
        )

    def run(
        self,
        example_input: Union[Tensor, np.ndarray, Dict],
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        compare_outputs: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Run benchmark comparison.

        Args:
            example_input: Example input for inference
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            compare_outputs: Compare output values
            rtol: Relative tolerance for output comparison
            atol: Absolute tolerance for output comparison

        Returns:
            Dictionary with benchmark results
        """
        results = {
            "pytorch": {},
            "onnx": {},
            "comparison": {},
        }

        # Prepare inputs
        if isinstance(example_input, Tensor):
            pytorch_input = example_input
            onnx_input = example_input.detach().cpu().numpy()
        elif isinstance(example_input, dict):
            pytorch_input = {k: v if isinstance(v, Tensor) else torch.tensor(v)
                           for k, v in example_input.items()}
            onnx_input = {k: v.detach().cpu().numpy() if isinstance(v, Tensor) else v
                        for k, v in example_input.items()}
        else:
            pytorch_input = torch.tensor(example_input)
            onnx_input = np.array(example_input)

        # Move to same device as model
        device = next(self.pytorch_model.parameters()).device
        if isinstance(pytorch_input, dict):
            pytorch_input = {k: v.to(device) for k, v in pytorch_input.items()}
        elif isinstance(pytorch_input, Tensor):
            pytorch_input = pytorch_input.to(device)

        # Benchmark PyTorch
        with torch.no_grad():
            # Warmup
            for _ in range(warmup_runs):
                if isinstance(pytorch_input, dict):
                    _ = self.pytorch_model(**pytorch_input)
                else:
                    _ = self.pytorch_model(pytorch_input)

            # Sync if CUDA
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            pytorch_latencies = []
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                if isinstance(pytorch_input, dict):
                    pytorch_output = self.pytorch_model(**pytorch_input)
                else:
                    pytorch_output = self.pytorch_model(pytorch_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                pytorch_latencies.append((end - start) * 1000)

        pytorch_latencies = np.array(pytorch_latencies)
        results["pytorch"] = {
            "mean_ms": float(np.mean(pytorch_latencies)),
            "std_ms": float(np.std(pytorch_latencies)),
            "min_ms": float(np.min(pytorch_latencies)),
            "max_ms": float(np.max(pytorch_latencies)),
            "p50_ms": float(np.percentile(pytorch_latencies, 50)),
            "p95_ms": float(np.percentile(pytorch_latencies, 95)),
            "p99_ms": float(np.percentile(pytorch_latencies, 99)),
        }

        # Benchmark ONNX Runtime
        onnx_results = self.onnx_inference.benchmark(onnx_input, warmup_runs, benchmark_runs)
        results["onnx"] = onnx_results

        # Get ONNX output for comparison
        onnx_output = self.onnx_inference.run(onnx_input)

        # Calculate speedup
        speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
        results["comparison"]["speedup"] = speedup
        results["comparison"]["onnx_faster"] = speedup > 1.0

        # Compare outputs
        if compare_outputs:
            pytorch_output_np = pytorch_output.detach().cpu().numpy() if isinstance(pytorch_output, Tensor) else pytorch_output
            onnx_output_np = onnx_output if isinstance(onnx_output, np.ndarray) else onnx_output[0]

            results["comparison"]["output_shape_match"] = pytorch_output_np.shape == onnx_output_np.shape
            results["comparison"]["output_close"] = np.allclose(
                pytorch_output_np, onnx_output_np, rtol=rtol, atol=atol
            )
            results["comparison"]["max_diff"] = float(np.max(np.abs(pytorch_output_np - onnx_output_np)))
            results["comparison"]["mean_diff"] = float(np.mean(np.abs(pytorch_output_np - onnx_output_np)))

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 60)
        print("Inference Benchmark Results")
        print("=" * 60)

        print("\nPyTorch Inference:")
        print(f"  Mean latency: {results['pytorch']['mean_ms']:.3f} ms")
        print(f"  Std latency:  {results['pytorch']['std_ms']:.3f} ms")
        print(f"  P50 latency:  {results['pytorch']['p50_ms']:.3f} ms")
        print(f"  P95 latency:  {results['pytorch']['p95_ms']:.3f} ms")
        print(f"  P99 latency:  {results['pytorch']['p99_ms']:.3f} ms")

        print("\nONNX Runtime Inference:")
        print(f"  Mean latency: {results['onnx']['mean_ms']:.3f} ms")
        print(f"  Std latency:  {results['onnx']['std_ms']:.3f} ms")
        print(f"  P50 latency:  {results['onnx']['p50_ms']:.3f} ms")
        print(f"  P95 latency:  {results['onnx']['p95_ms']:.3f} ms")
        print(f"  P99 latency:  {results['onnx']['p99_ms']:.3f} ms")

        print("\nComparison:")
        speedup = results["comparison"]["speedup"]
        if speedup > 1:
            print(f"  ONNX Runtime is {speedup:.2f}x FASTER")
        else:
            print(f"  PyTorch is {1/speedup:.2f}x faster")

        if "output_close" in results["comparison"]:
            print(f"  Outputs match: {results['comparison']['output_close']}")
            print(f"  Max difference: {results['comparison']['max_diff']:.6f}")
            print(f"  Mean difference: {results['comparison']['mean_diff']:.6f}")

        print("=" * 60)


# =============================================================================
# Convenience Functions
# =============================================================================


def load_onnx_model(
    model_path: str,
    execution_provider: str = "cpu",
    **kwargs
) -> ONNXInference:
    """
    Load an ONNX model for inference.

    Args:
        model_path: Path to ONNX model
        execution_provider: "cpu", "cuda", or "tensorrt"
        **kwargs: Additional InferenceConfig arguments

    Returns:
        ONNXInference instance
    """
    provider_map = {
        "cpu": ExecutionProvider.CPU,
        "cuda": ExecutionProvider.CUDA,
        "tensorrt": ExecutionProvider.TENSORRT,
        "gpu": ExecutionProvider.CUDA,
    }

    config = InferenceConfig(
        execution_provider=provider_map.get(execution_provider.lower(), ExecutionProvider.CPU),
        **kwargs
    )

    return ONNXInference(model_path, config)


def benchmark_pytorch_vs_onnx(
    pytorch_model: "nn.Module",
    onnx_model_path: str,
    example_input: Tensor,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, Any]:
    """
    Quick benchmark comparison between PyTorch and ONNX.

    Args:
        pytorch_model: PyTorch model
        onnx_model_path: Path to ONNX model
        example_input: Example input tensor
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs

    Returns:
        Benchmark results dictionary
    """
    benchmark = InferenceBenchmark(pytorch_model, onnx_model_path)
    results = benchmark.run(example_input, warmup_runs, benchmark_runs)
    benchmark.print_results(results)
    return results


# =============================================================================
# Registry
# =============================================================================


ONNX_INFERENCE_COMPONENTS = {
    "enums": ["ExecutionProvider", "GraphOptimization"],
    "config": ["InferenceConfig"],
    "inference": ["ONNXInference"],
    "benchmark": ["InferenceBenchmark"],
    "functions": ["load_onnx_model", "benchmark_pytorch_vs_onnx"],
}
