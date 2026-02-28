"""
Knowledge Distillation Experiments for Model Compression Comparison.

This module provides:
    - DistillationExperiment: End-to-end distillation experiments
    - CompressionComparison: Compare different compression methods
    - TeacherStudentConfig: Configuration for teacher-student pairs
    - ExperimentReport: Generate experiment reports

Theory:
    This module provides experimental frameworks to evaluate knowledge
    distillation effectiveness by comparing:
    - Teacher vs Student accuracy
    - Model size reduction
    - Inference speedup
    - Training efficiency

    Key Metrics:
    - Accuracy Retention: student_accuracy / teacher_accuracy
    - Compression Ratio: teacher_params / student_params
    - Speedup: teacher_latency / student_latency
    - Knowledge Transfer Efficiency: (student_acc - baseline) / (teacher_acc - baseline)

References:
    - Model Compression Survey: https://arxiv.org/abs/2010.05058
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import copy
import json
import warnings

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    Dataset = None
    Tensor = None

from .distillation import (
    KnowledgeDistiller,
    DistillationConfig,
    DistillationType,
    TemperatureSchedule,
    create_distiller,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Configuration
# =============================================================================


@dataclass
class TeacherStudentConfig:
    """
    Configuration for teacher-student model pair.

    Attributes:
        teacher_model: Teacher model factory or instance
        student_model: Student model factory or instance
        teacher_checkpoint: Path to teacher checkpoint (optional)
        teacher_name: Name of teacher architecture
        student_name: Name of student architecture
        input_shape: Input tensor shape (without batch)
        num_classes: Number of output classes
    """
    teacher_model: Any = None
    student_model: Any = None
    teacher_checkpoint: Optional[str] = None
    teacher_name: str = "teacher"
    student_name: str = "student"
    input_shape: Tuple[int, ...] = (3, 224, 224)
    num_classes: int = 1000


@dataclass
class ExperimentConfig:
    """
    Configuration for distillation experiments.

    Attributes:
        num_epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        temperatures: List of temperatures to try
        alphas: List of alpha values to try
        early_stopping_patience: Early stopping patience
        save_checkpoints: Whether to save model checkpoints
        checkpoint_dir: Directory for checkpoints
        log_interval: Batches between logging
        seed: Random seed for reproducibility
    """
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperatures: List[float] = field(default_factory=lambda: [4.0])
    alphas: List[float] = field(default_factory=lambda: [0.3])
    early_stopping_patience: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    seed: int = 42


# =============================================================================
# Experiment Results
# =============================================================================


@dataclass
class DistillationResult:
    """Results from a single distillation experiment."""
    temperature: float
    alpha: float
    teacher_accuracy: float
    student_accuracy: float
    accuracy_retention: float  # student/teacher ratio
    teacher_params: int
    student_params: int
    compression_ratio: float
    teacher_latency_ms: float
    student_latency_ms: float
    speedup: float
    training_time_sec: float
    final_train_loss: float
    best_val_loss: float
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "alpha": self.alpha,
            "teacher_accuracy": self.teacher_accuracy,
            "student_accuracy": self.student_accuracy,
            "accuracy_retention": self.accuracy_retention,
            "teacher_params": self.teacher_params,
            "student_params": self.student_params,
            "compression_ratio": self.compression_ratio,
            "teacher_latency_ms": self.teacher_latency_ms,
            "student_latency_ms": self.student_latency_ms,
            "speedup": self.speedup,
            "training_time_sec": self.training_time_sec,
            "final_train_loss": self.final_train_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }


@dataclass
class ExperimentReport:
    """Complete experiment report."""
    experiment_name: str
    teacher_name: str
    student_name: str
    results: List[DistillationResult]
    best_result: Optional[DistillationResult]
    baseline_accuracy: float
    total_time_sec: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "teacher_name": self.teacher_name,
            "student_name": self.student_name,
            "results": [r.to_dict() for r in self.results],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "baseline_accuracy": self.baseline_accuracy,
            "total_time_sec": self.total_time_sec,
            "timestamp": self.timestamp,
        }

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Teacher: {self.teacher_name}",
            f"Student: {self.student_name}",
            f"Baseline Accuracy: {self.baseline_accuracy:.4f}",
            "",
            "Best Result:",
        ]
        if self.best_result:
            lines.extend([
                f"  Temperature: {self.best_result.temperature}",
                f"  Alpha: {self.best_result.alpha}",
                f"  Student Accuracy: {self.best_result.student_accuracy:.4f}",
                f"  Accuracy Retention: {self.best_result.accuracy_retention:.2%}",
                f"  Compression Ratio: {self.best_result.compression_ratio:.2f}x",
                f"  Speedup: {self.best_result.speedup:.2f}x",
            ])
        lines.append(f"\nTotal Experiments: {len(self.results)}")
        lines.append(f"Total Time: {self.total_time_sec:.2f}s")
        return "\n".join(lines)


# =============================================================================
# Distillation Experiment
# =============================================================================


class DistillationExperiment:
    """
    End-to-end knowledge distillation experiment.

    This class runs complete distillation experiments including:
    - Hyperparameter search (temperature, alpha)
    - Baseline comparison
    - Performance benchmarking
    - Result reporting

    Example:
        >>> experiment = DistillationExperiment(
        ...     teacher_model=resnet50,
        ...     student_model=resnet18,
        ...     config=ExperimentConfig()
        ... )
        >>> report = experiment.run(train_loader, val_loader, test_loader)
        >>> print(report.summary())
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        experiment_config: Optional[ExperimentConfig] = None,
        model_config: Optional[TeacherStudentConfig] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = "distillation_experiment",
    ):
        """
        Initialize distillation experiment.

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            experiment_config: Experiment configuration
            model_config: Model pair configuration
            device: Device to use
            experiment_name: Name for this experiment
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for distillation experiments")

        self.experiment_config = experiment_config or ExperimentConfig()
        self.model_config = model_config or TeacherStudentConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name

        # Store model references
        self.teacher_model = teacher_model.to(self.device)
        self.student_model_template = copy.deepcopy(student_model)
        self.student_model = student_model.to(self.device)

        # Set random seed
        torch.manual_seed(self.experiment_config.seed)
        np.random.seed(self.experiment_config.seed)

        # Create checkpoint directory
        if self.experiment_config.save_checkpoints:
            os.makedirs(self.experiment_config.checkpoint_dir, exist_ok=True)

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        baseline_loader: Optional[DataLoader] = None,
    ) -> ExperimentReport:
        """
        Run complete distillation experiment.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (uses val_loader if None)
            baseline_loader: Loader for baseline evaluation

        Returns:
            Complete experiment report
        """
        start_time = time.time()
        results = []

        # Evaluate teacher
        teacher_accuracy = self._evaluate_model(self.teacher_model, test_loader or val_loader)
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())

        # Get baseline accuracy (student trained without distillation)
        baseline_accuracy = self._get_baseline_accuracy(train_loader, val_loader)

        # Run experiments for each temperature and alpha combination
        for temp in self.experiment_config.temperatures:
            for alpha in self.experiment_config.alphas:
                logger.info(f"Running experiment: T={temp}, α={alpha}")

                result = self._run_single_experiment(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    temperature=temp,
                    alpha=alpha,
                    teacher_accuracy=teacher_accuracy,
                )
                results.append(result)

        # Find best result
        best_result = max(results, key=lambda r: r.student_accuracy)

        total_time = time.time() - start_time

        report = ExperimentReport(
            experiment_name=self.experiment_name,
            teacher_name=self.model_config.teacher_name,
            student_name=self.model_config.student_name,
            results=results,
            best_result=best_result,
            baseline_accuracy=baseline_accuracy,
            total_time_sec=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        return report

    def _run_single_experiment(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        temperature: float,
        alpha: float,
        teacher_accuracy: float,
    ) -> DistillationResult:
        """Run a single distillation experiment."""
        # Reset student model
        self.student_model = copy.deepcopy(self.student_model_template).to(self.device)

        # Create distillation config
        config = DistillationConfig(
            temperature=temperature,
            alpha=alpha,
        )

        # Create distiller
        distiller = KnowledgeDistiller(
            self.teacher_model,
            self.student_model,
            config,
            self.device,
        )

        # Train
        start_time = time.time()
        optimizer = optim.Adam(
            self.student_model.parameters(),
            lr=self.experiment_config.learning_rate,
            weight_decay=self.experiment_config.weight_decay,
        )

        history = distiller.train(
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=self.experiment_config.num_epochs,
            val_loader=val_loader,
        )
        training_time = time.time() - start_time

        # Load best model
        distiller.load_best_model()

        # Evaluate
        test_loader = test_loader or val_loader
        metrics = distiller.evaluate(test_loader)

        # Get speed metrics
        speed_metrics = distiller.compare_inference_speed(
            self.model_config.input_shape,
        )

        # Create result
        result = DistillationResult(
            temperature=temperature,
            alpha=alpha,
            teacher_accuracy=teacher_accuracy,
            student_accuracy=metrics["accuracy"],
            accuracy_retention=metrics["accuracy"] / teacher_accuracy if teacher_accuracy > 0 else 0,
            teacher_params=speed_metrics["teacher_params"],
            student_params=speed_metrics["student_params"],
            compression_ratio=speed_metrics["compression_ratio"],
            teacher_latency_ms=speed_metrics["teacher_latency_ms"],
            student_latency_ms=speed_metrics["student_latency_ms"],
            speedup=speed_metrics["speedup"],
            training_time_sec=training_time,
            final_train_loss=history["final_train_loss"],
            best_val_loss=distiller.best_loss,
            config={
                "num_epochs": self.experiment_config.num_epochs,
                "learning_rate": self.experiment_config.learning_rate,
            },
        )

        return result

    def _get_baseline_accuracy(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> float:
        """Train student without distillation to get baseline."""
        # Reset student model
        student_model = copy.deepcopy(self.student_model_template).to(self.device)
        student_model.train()

        optimizer = optim.Adam(
            student_model.parameters(),
            lr=self.experiment_config.learning_rate,
            weight_decay=self.experiment_config.weight_decay,
        )

        ce_loss = nn.CrossEntropyLoss()

        # Quick training (fewer epochs for baseline)
        num_epochs = max(5, self.experiment_config.num_epochs // 2)
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = student_model(inputs)
                loss = ce_loss(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate
        return self._evaluate_model(student_model, val_loader)

    @torch.no_grad()
    def _evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        return correct / total if total > 0 else 0.0


# =============================================================================
# Compression Comparison
# =============================================================================


class CompressionComparison:
    """
    Compare different model compression methods.

    This class provides utilities to compare:
    - Knowledge distillation
    - Pruning
    - Quantization
    - Combination of methods

    Example:
        >>> comparison = CompressionComparison(teacher_model)
        >>> results = comparison.compare_all(train_loader, val_loader)
        >>> comparison.print_comparison(results)
    """

    def __init__(
        self,
        original_model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize compression comparison.

        Args:
            original_model: Original uncompressed model
            device: Device to use
        """
        self.original_model = original_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_model.to(self.device)

    def compare_distillation(
        self,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        temperatures: List[float] = [4.0],
        num_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare distillation results.

        Args:
            student_model: Student model for distillation
            train_loader: Training data
            val_loader: Validation data
            temperatures: Temperatures to try
            num_epochs: Training epochs

        Returns:
            Comparison results
        """
        results = []

        for temp in temperatures:
            config = DistillationConfig(temperature=temp)
            student = copy.deepcopy(student_model).to(self.device)

            distiller = KnowledgeDistiller(
                self.original_model, student, config, self.device
            )

            distiller.train(train_loader, num_epochs=num_epochs)
            distiller.load_best_model()

            metrics = distiller.evaluate(val_loader)
            speed = distiller.compare_inference_speed((3, 32, 32))

            results.append({
                "temperature": temp,
                "accuracy": metrics["accuracy"],
                "speedup": speed["speedup"],
                "compression_ratio": speed["compression_ratio"],
            })

        return {
            "method": "distillation",
            "results": results,
            "best": max(results, key=lambda x: x["accuracy"]),
        }

    def print_comparison(self, results: Dict[str, Any]) -> None:
        """Print comparison results in formatted table."""
        print("\n" + "=" * 70)
        print("Model Compression Comparison")
        print("=" * 70)

        for method, data in results.items():
            if isinstance(data, dict) and "results" in data:
                print(f"\n{method.upper()}:")
                for r in data["results"]:
                    print(f"  Config: {r.get('temperature', r.get('sparsity', 'N/A'))}")
                    print(f"    Accuracy: {r['accuracy']:.4f}")
                    print(f"    Speedup: {r['speedup']:.2f}x")
                    print(f"    Compression: {r['compression_ratio']:.2f}x")

        print("\n" + "=" * 70)


# =============================================================================
# Convenience Functions
# =============================================================================


def run_distillation_experiment(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    temperatures: List[float] = [2.0, 4.0, 8.0],
    alphas: List[float] = [0.1, 0.3, 0.5],
) -> ExperimentReport:
    """
    Quick function to run distillation experiment.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        train_loader: Training data
        val_loader: Validation data
        num_epochs: Training epochs
        temperatures: Temperatures to try
        alphas: Alpha values to try

    Returns:
        Experiment report
    """
    config = ExperimentConfig(
        num_epochs=num_epochs,
        temperatures=temperatures,
        alphas=alphas,
    )

    experiment = DistillationExperiment(
        teacher_model=teacher_model,
        student_model=student_model,
        experiment_config=config,
    )

    return experiment.run(train_loader, val_loader)


def compare_teacher_student(
    teacher_model: nn.Module,
    student_model: nn.Module,
    data_loader: DataLoader,
    input_shape: Tuple[int, ...] = (3, 32, 32),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Quick comparison between teacher and student models.

    Args:
        teacher_model: Teacher model
        student_model: Student model
        data_loader: Data for evaluation
        input_shape: Input shape for speed benchmark
        device: Device to use

    Returns:
        Comparison metrics
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    teacher_model.eval()
    student_model.eval()

    # Evaluate accuracy
    def evaluate(model):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(-1) == targets).sum().item()
                total += targets.size(0)
        return correct / total

    teacher_acc = evaluate(teacher_model)
    student_acc = evaluate(student_model)

    # Benchmark speed
    dummy_input = torch.randn(1, *input_shape, device=device)

    def benchmark(model, num_runs=100):
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return np.mean(times)

    teacher_latency = benchmark(teacher_model)
    student_latency = benchmark(student_model)

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    return {
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
        "accuracy_retention": student_acc / teacher_acc if teacher_acc > 0 else 0,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / max(student_params, 1),
        "teacher_latency_ms": teacher_latency,
        "student_latency_ms": student_latency,
        "speedup": teacher_latency / student_latency if student_latency > 0 else 0,
    }


# =============================================================================
# Registry
# =============================================================================


DISTILLATION_EXPERIMENTS_COMPONENTS = {
    "config": ["TeacherStudentConfig", "ExperimentConfig"],
    "results": ["DistillationResult", "ExperimentReport"],
    "experiments": ["DistillationExperiment"],
    "comparison": ["CompressionComparison"],
    "functions": ["run_distillation_experiment", "compare_teacher_student"],
}
