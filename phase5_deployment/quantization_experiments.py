"""
Quantization Experiments for Accuracy and Speed Comparison.

This module provides:
    - QuantizationTypeComparison: Compare dynamic/static/QAT
    - PrecisionComparison: Compare INT8/INT4/FP16
    - CalibrationExperiment: Test calibration data requirements
    - InferenceBenchmark: Measure inference speedup

Experiments:
    1. Type comparison: Dynamic vs Static vs QAT
    2. Precision comparison: INT8 vs INT4 vs FP16
    3. Accuracy vs compression tradeoff
    4. Calibration sensitivity analysis
    5. Inference speedup benchmark

Usage:
    from phase5_deployment.quantization_experiments import (
        QuantizationTypeComparison,
        PrecisionComparisonExperiment,
        InferenceBenchmark,
        run_all_quantization_experiments
    )
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
import copy
import time
import logging
import json
from pathlib import Path

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    Subset = None

from .quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizationDtype,
    QuantizationManager,
    DynamicQuantizer,
    StaticQuantizer,
    QATQuantizer,
    INT4Quantizer,
    BaseQuantizer,
    create_quantizer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Result Data Classes
# =============================================================================


@dataclass
class QuantizationResult:
    """Result from a quantization experiment."""
    qtype: str
    dtype: str
    accuracy: float
    accuracy_drop: float
    model_size_mb: float
    compression_ratio: float
    inference_time_ms: float
    speedup: float


@dataclass
class CalibrationResult:
    """Result from calibration experiment."""
    num_calibration_batches: int
    accuracy: float
    calibration_time_s: float


@dataclass
class ExperimentReport:
    """Complete experiment report."""
    experiment_name: str
    description: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'results': self.results,
            'summary': self.summary,
            'timestamp': self.timestamp
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Evaluation Utilities
# =============================================================================


def evaluate_model(
    model: "nn.Module",
    dataloader: "DataLoader",
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model accuracy.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use

    Returns:
        Dictionary with accuracy and loss
    """
    model.eval()
    model.to(device)

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

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)

    return {'accuracy': accuracy, 'loss': avg_loss}


def measure_inference_time(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    n_runs: int = 100,
    warmup: int = 10
) -> float:
    """
    Measure average inference time.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch)
        device: Device to use
        n_runs: Number of runs
        warmup: Warmup runs

    Returns:
        Average inference time in milliseconds
    """
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


# =============================================================================
# Quantization Type Comparison Experiment
# =============================================================================


class QuantizationTypeComparison:
    """
    Compare different quantization types.

    Compares:
        - Dynamic quantization
        - Static quantization (PTQ)
        - Quantization-Aware Training (QAT)
    """

    TYPES = [QuantizationType.DYNAMIC, QuantizationType.STATIC, QuantizationType.QAT]

    def __init__(
        self,
        types: Optional[List[QuantizationType]] = None,
        dtype: QuantizationDtype = QuantizationDtype.INT8
    ):
        """
        Initialize type comparison.

        Args:
            types: Types to compare
            dtype: Target data type
        """
        self.types = types or self.TYPES
        self.dtype = dtype
        self.results: List[QuantizationResult] = []

    def run(
        self,
        model: "nn.Module",
        eval_loader: "DataLoader",
        calibration_loader: Optional["DataLoader"] = None,
        train_loader: Optional["DataLoader"] = None,
        val_loader: Optional["DataLoader"] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        device: str = 'cpu'
    ) -> List[QuantizationResult]:
        """
        Run type comparison experiment.

        Args:
            model: PyTorch model
            eval_loader: Evaluation data
            calibration_loader: Calibration data for static
            train_loader: Training data for QAT
            val_loader: Validation data for QAT
            input_shape: Input shape for timing
            device: Device to use

        Returns:
            List of QuantizationResult
        """
        # Evaluate original model
        original_eval = evaluate_model(model, eval_loader, device)
        original_accuracy = original_eval['accuracy']

        base = BaseQuantizer()
        original_size = base.get_model_size_mb(model)
        original_time = 0.0
        if input_shape:
            original_time = measure_inference_time(model, input_shape, device)

        self.results = []

        for qtype in self.types:
            config = QuantizationConfig(qtype=qtype, dtype=self.dtype)
            manager = QuantizationManager(config)
            manager.save_original_model(model)

            try:
                if qtype == QuantizationType.DYNAMIC:
                    quantized = manager.quantize(model, device=device)
                elif qtype == QuantizationType.STATIC:
                    if calibration_loader is None:
                        logger.warning(f"Skipping {qtype.value}: no calibration data")
                        continue
                    quantized = manager.quantize(model, calibration_loader=calibration_loader, device=device)
                elif qtype == QuantizationType.QAT:
                    if train_loader is None:
                        logger.warning(f"Skipping {qtype.value}: no training data")
                        continue
                    quantized = manager.quantize(model, train_loader=train_loader, val_loader=val_loader, device=device)

                # Evaluate
                quantized_eval = evaluate_model(quantized, eval_loader, device)
                accuracy = quantized_eval['accuracy']

                # Measure size
                quantized_size = base.get_model_size_mb(quantized)
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

                # Measure time
                quantized_time = 0.0
                if input_shape:
                    quantized_time = measure_inference_time(quantized, input_shape, device)

                result = QuantizationResult(
                    qtype=qtype.value,
                    dtype=self.dtype.value,
                    accuracy=accuracy,
                    accuracy_drop=original_accuracy - accuracy,
                    model_size_mb=quantized_size,
                    compression_ratio=compression_ratio,
                    inference_time_ms=quantized_time,
                    speedup=original_time / quantized_time if quantized_time > 0 else 1.0
                )

                self.results.append(result)
                logger.info(f"{qtype.value}: accuracy={accuracy:.4f}, "
                           f"drop={result.accuracy_drop:.4f}, "
                           f"compression={compression_ratio:.2f}x")

            except Exception as e:
                logger.error(f"Error with {qtype.value}: {e}")

        return self.results

    def get_report(self) -> ExperimentReport:
        """Generate experiment report."""
        results_list = [
            {
                'qtype': r.qtype,
                'dtype': r.dtype,
                'accuracy': r.accuracy,
                'accuracy_drop': r.accuracy_drop,
                'model_size_mb': r.model_size_mb,
                'compression_ratio': r.compression_ratio,
                'inference_time_ms': r.inference_time_ms,
                'speedup': r.speedup
            }
            for r in self.results
        ]

        # Find best method
        best = min(self.results, key=lambda r: r.accuracy_drop) if self.results else None

        summary = {
            'dtype': self.dtype.value,
            'types_compared': [r.qtype for r in self.results],
            'best_for_accuracy': best.qtype if best else None,
            'best_accuracy_drop': best.accuracy_drop if best else None,
        }

        return ExperimentReport(
            experiment_name="Quantization Type Comparison",
            description=f"Compare dynamic, static, and QAT quantization with {self.dtype.value}",
            results=results_list,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# =============================================================================
# Precision Comparison Experiment
# =============================================================================


class PrecisionComparisonExperiment:
    """
    Compare different quantization precisions.

    Compares:
        - INT8
        - INT4
        - FP16
    """

    PRECISIONS = [QuantizationDtype.INT8, QuantizationDtype.INT4, QuantizationDtype.FP16]

    def __init__(
        self,
        precisions: Optional[List[QuantizationDtype]] = None,
        qtype: QuantizationType = QuantizationType.STATIC
    ):
        """
        Initialize precision comparison.

        Args:
            precisions: Precisions to compare
            qtype: Quantization type to use
        """
        self.precisions = precisions or self.PRECISIONS
        self.qtype = qtype
        self.results: List[QuantizationResult] = []

    def run(
        self,
        model: "nn.Module",
        eval_loader: "DataLoader",
        calibration_loader: Optional["DataLoader"] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        device: str = 'cpu'
    ) -> List[QuantizationResult]:
        """
        Run precision comparison.

        Args:
            model: PyTorch model
            eval_loader: Evaluation data
            calibration_loader: Calibration data
            input_shape: Input shape for timing
            device: Device to use

        Returns:
            List of QuantizationResult
        """
        # Evaluate original
        original_eval = evaluate_model(model, eval_loader, device)
        original_accuracy = original_eval['accuracy']

        base = BaseQuantizer()
        original_size = base.get_model_size_mb(model)
        original_time = 0.0
        if input_shape:
            original_time = measure_inference_time(model, input_shape, device)

        self.results = []

        for dtype in self.precisions:
            config = QuantizationConfig(qtype=self.qtype, dtype=dtype)
            manager = QuantizationManager(config)
            manager.save_original_model(model)

            try:
                quantized = manager.quantize(model, calibration_loader=calibration_loader, device=device)

                # Evaluate
                quantized_eval = evaluate_model(quantized, eval_loader, device)
                accuracy = quantized_eval['accuracy']

                # Measure
                quantized_size = base.get_model_size_mb(quantized)
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

                quantized_time = 0.0
                if input_shape:
                    quantized_time = measure_inference_time(quantized, input_shape, device)

                result = QuantizationResult(
                    qtype=self.qtype.value,
                    dtype=dtype.value,
                    accuracy=accuracy,
                    accuracy_drop=original_accuracy - accuracy,
                    model_size_mb=quantized_size,
                    compression_ratio=compression_ratio,
                    inference_time_ms=quantized_time,
                    speedup=original_time / quantized_time if quantized_time > 0 else 1.0
                )

                self.results.append(result)
                logger.info(f"{dtype.value}: accuracy={accuracy:.4f}, "
                           f"compression={compression_ratio:.2f}x")

            except Exception as e:
                logger.error(f"Error with {dtype.value}: {e}")

        return self.results

    def get_report(self) -> ExperimentReport:
        """Generate experiment report."""
        results_list = [
            {
                'dtype': r.dtype,
                'accuracy': r.accuracy,
                'accuracy_drop': r.accuracy_drop,
                'compression_ratio': r.compression_ratio,
            }
            for r in self.results
        ]

        summary = {
            'qtype': self.qtype.value,
            'precisions_compared': [r.dtype for r in self.results],
            'best_compression': max(self.results, key=lambda r: r.compression_ratio).dtype if self.results else None,
        }

        return ExperimentReport(
            experiment_name="Precision Comparison",
            description=f"Compare INT8, INT4, FP16 with {self.qtype.value} quantization",
            results=results_list,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# =============================================================================
# Calibration Experiment
# =============================================================================


class CalibrationExperiment:
    """
    Test calibration data requirements.

    Varies number of calibration batches and measures accuracy.
    """

    def __init__(
        self,
        batch_counts: Optional[List[int]] = None
    ):
        """
        Initialize calibration experiment.

        Args:
            batch_counts: List of calibration batch counts to test
        """
        self.batch_counts = batch_counts or [1, 5, 10, 20, 50, 100]
        self.results: List[CalibrationResult] = []

    def run(
        self,
        model: "nn.Module",
        eval_loader: "DataLoader",
        calibration_loader: "DataLoader",
        device: str = 'cpu'
    ) -> List[CalibrationResult]:
        """
        Run calibration experiment.

        Args:
            model: PyTorch model
            eval_loader: Evaluation data
            calibration_loader: Calibration data
            device: Device to use

        Returns:
            List of CalibrationResult
        """
        self.results = []

        for n_batches in self.batch_counts:
            config = QuantizationConfig(
                qtype=QuantizationType.STATIC,
                calibration_batches=n_batches
            )

            try:
                start_time = time.perf_counter()
                manager = QuantizationManager(config)
                quantized = manager.quantize(model, calibration_loader=calibration_loader, device=device)
                calibration_time = time.perf_counter() - start_time

                # Evaluate
                eval_result = evaluate_model(quantized, eval_loader, device)

                result = CalibrationResult(
                    num_calibration_batches=n_batches,
                    accuracy=eval_result['accuracy'],
                    calibration_time_s=calibration_time
                )

                self.results.append(result)
                logger.info(f"{n_batches} batches: accuracy={eval_result['accuracy']:.4f}, "
                           f"time={calibration_time:.2f}s")

            except Exception as e:
                logger.error(f"Error with {n_batches} batches: {e}")

        return self.results

    def get_report(self) -> ExperimentReport:
        """Generate experiment report."""
        results_list = [
            {
                'num_calibration_batches': r.num_calibration_batches,
                'accuracy': r.accuracy,
                'calibration_time_s': r.calibration_time_s
            }
            for r in self.results
        ]

        # Find optimal calibration count
        # (good accuracy with minimal calibration)
        optimal = None
        if self.results:
            # Find where accuracy plateaus
            best_acc = max(r.accuracy for r in self.results)
            for r in sorted(self.results, key=lambda x: x.num_calibration_batches):
                if r.accuracy >= best_acc * 0.99:  # Within 1% of best
                    optimal = r.num_calibration_batches
                    break

        summary = {
            'batch_counts_tested': self.batch_counts,
            'best_accuracy': max(r.accuracy for r in self.results) if self.results else None,
            'optimal_calibration_batches': optimal,
        }

        return ExperimentReport(
            experiment_name="Calibration Sensitivity",
            description="Test accuracy vs number of calibration batches",
            results=results_list,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# =============================================================================
# Inference Benchmark
# =============================================================================


class InferenceBenchmark:
    """
    Benchmark inference speedup from quantization.

    Measures:
        - Original model inference time
        - Quantized model inference time
        - Speedup ratio
    """

    def __init__(
        self,
        n_runs: int = 100,
        warmup: int = 10
    ):
        """
        Initialize benchmark.

        Args:
            n_runs: Number of inference runs
            warmup: Warmup runs
        """
        self.n_runs = n_runs
        self.warmup = warmup

    def run(
        self,
        original_model: "nn.Module",
        quantized_model: "nn.Module",
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Run inference benchmark.

        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            input_shape: Input tensor shape
            device: Device to use

        Returns:
            Benchmark results
        """
        original_time = measure_inference_time(
            original_model, input_shape, device, self.n_runs, self.warmup
        )

        quantized_time = measure_inference_time(
            quantized_model, input_shape, device, self.n_runs, self.warmup
        )

        speedup = original_time / quantized_time if quantized_time > 0 else 1.0

        return {
            'original_time_ms': original_time,
            'quantized_time_ms': quantized_time,
            'speedup': speedup,
            'n_runs': self.n_runs,
            'device': device
        }


# =============================================================================
# Complete Quantization Experiment
# =============================================================================


class CompleteQuantizationExperiment:
    """
    Complete quantization experiment pipeline.

    Runs all experiments:
        1. Type comparison
        2. Precision comparison
        3. Calibration sensitivity
        4. Inference benchmark
    """

    def __init__(
        self,
        config: Optional[QuantizationConfig] = None
    ):
        """
        Initialize complete experiment.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()

    def run(
        self,
        model: "nn.Module",
        eval_loader: "DataLoader",
        calibration_loader: Optional["DataLoader"] = None,
        train_loader: Optional["DataLoader"] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Run complete quantization experiment.

        Args:
            model: PyTorch model
            eval_loader: Evaluation data
            calibration_loader: Calibration data
            train_loader: Training data
            input_shape: Input shape for timing
            device: Device to use

        Returns:
            Complete experiment results
        """
        results = {}

        # 1. Evaluate original
        logger.info("Evaluating original model...")
        original_eval = evaluate_model(model, eval_loader, device)
        results['original'] = original_eval

        base = BaseQuantizer()
        results['original']['size_mb'] = base.get_model_size_mb(model)

        # 2. Type comparison
        logger.info("Running type comparison...")
        type_comparison = QuantizationTypeComparison()
        type_comparison.run(
            model, eval_loader,
            calibration_loader=calibration_loader,
            train_loader=train_loader,
            input_shape=input_shape,
            device=device
        )
        results['type_comparison'] = type_comparison.get_report().to_dict()

        # 3. Precision comparison (using static quantization)
        logger.info("Running precision comparison...")
        precision_comparison = PrecisionComparisonExperiment(qtype=QuantizationType.STATIC)
        precision_comparison.run(
            model, eval_loader,
            calibration_loader=calibration_loader,
            input_shape=input_shape,
            device=device
        )
        results['precision_comparison'] = precision_comparison.get_report().to_dict()

        # 4. Calibration sensitivity
        if calibration_loader is not None:
            logger.info("Running calibration experiment...")
            calibration_exp = CalibrationExperiment()
            calibration_exp.run(model, eval_loader, calibration_loader, device)
            results['calibration'] = calibration_exp.get_report().to_dict()

        # 5. Summary
        results['summary'] = {
            'original_accuracy': original_eval['accuracy'],
            'original_size_mb': results['original']['size_mb'],
        }

        return results


# =============================================================================
# Run All Experiments
# =============================================================================


def run_all_quantization_experiments(
    model: "nn.Module",
    eval_loader: "DataLoader",
    calibration_loader: Optional["DataLoader"] = None,
    train_loader: Optional["DataLoader"] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    device: str = 'cpu',
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, ExperimentReport]:
    """
    Run all quantization experiments.

    Args:
        model: PyTorch model
        eval_loader: Evaluation data
        calibration_loader: Calibration data
        train_loader: Training data
        input_shape: Input shape
        device: Device to use
        output_dir: Directory to save reports

    Returns:
        Dictionary of experiment reports
    """
    reports = {}

    # 1. Type comparison
    logger.info("=" * 50)
    logger.info("Running Quantization Type Comparison...")
    type_exp = QuantizationTypeComparison()
    type_exp.run(model, eval_loader, calibration_loader=calibration_loader,
                 train_loader=train_loader, input_shape=input_shape, device=device)
    reports['type_comparison'] = type_exp.get_report()

    # 2. Precision comparison
    logger.info("=" * 50)
    logger.info("Running Precision Comparison...")
    prec_exp = PrecisionComparisonExperiment()
    prec_exp.run(model, eval_loader, calibration_loader=calibration_loader,
                 input_shape=input_shape, device=device)
    reports['precision_comparison'] = prec_exp.get_report()

    # 3. Calibration sensitivity
    if calibration_loader is not None:
        logger.info("=" * 50)
        logger.info("Running Calibration Experiment...")
        cal_exp = CalibrationExperiment()
        cal_exp.run(model, eval_loader, calibration_loader, device)
        reports['calibration'] = cal_exp.get_report()

    # Save reports
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, report in reports.items():
            report.save(output_dir / f"{name}.json")

    return reports


# =============================================================================
# Registry
# =============================================================================


QUANTIZATION_EXPERIMENTS_COMPONENTS = {
    'QuantizationResult': QuantizationResult,
    'CalibrationResult': CalibrationResult,
    'ExperimentReport': ExperimentReport,
    'QuantizationTypeComparison': QuantizationTypeComparison,
    'PrecisionComparisonExperiment': PrecisionComparisonExperiment,
    'CalibrationExperiment': CalibrationExperiment,
    'InferenceBenchmark': InferenceBenchmark,
    'CompleteQuantizationExperiment': CompleteQuantizationExperiment,
    'evaluate_model': evaluate_model,
    'measure_inference_time': measure_inference_time,
    'run_all_quantization_experiments': run_all_quantization_experiments,
}
