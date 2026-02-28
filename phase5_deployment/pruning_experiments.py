"""
Pruning Experiments for Model Compression Comparison.

This module provides:
    - Compression ratio benchmarks
    - Accuracy vs sparsity tradeoff analysis
    - Fine-tuning pipeline after pruning
    - Pruning method comparison experiments

Experiments:
    1. Sparsity sweep: Vary sparsity 0% to 90%, measure accuracy
    2. Method comparison: Compare magnitude, random, gradient, global
    3. Structured vs unstructured: Compare channel vs magnitude pruning
    4. Fine-tuning recovery: Measure accuracy recovery after fine-tuning
    5. Iterative vs one-shot: Compare iterative vs one-shot pruning

Usage:
    from phase5_deployment.pruning_experiments import (
        SparsitySweepExperiment,
        MethodComparisonExperiment,
        FineTuningPipeline,
        run_all_experiments
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
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Subset
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None
    Subset = None
    autocast = None
    GradScaler = None

from .pruning import (
    PruningConfig,
    PruningMethod,
    PruningManager,
    MagnitudePruner,
    RandomPruner,
    GlobalPruner,
    ChannelPruner,
    IterativePruningSchedule,
    create_pruner,
    get_model_sparsity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Result Data Classes
# =============================================================================


@dataclass
class SparsityResult:
    """Result from a single sparsity level experiment."""
    sparsity: float
    accuracy: float
    accuracy_drop: float
    model_size_mb: float
    compression_ratio: float
    inference_time_ms: float


@dataclass
class MethodResult:
    """Result from comparing pruning methods."""
    method_name: str
    sparsity: float
    accuracy: float
    accuracy_drop: float
    model_size_mb: float
    compression_ratio: float
    fine_tuned_accuracy: Optional[float] = None


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
    device: str = 'cuda',
    criterion: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use
        criterion: Loss function (optional)

    Returns:
        Dictionary with accuracy and optional loss
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if criterion is not None:
                total_loss += criterion(outputs, targets).item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if criterion is not None else None

    result = {'accuracy': accuracy}
    if avg_loss is not None:
        result['loss'] = avg_loss

    return result


def measure_inference_time(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    n_runs: int = 100,
    warmup: int = 10
) -> float:
    """
    Measure average inference time.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        device: Device to use
        n_runs: Number of inference runs
        warmup: Number of warmup runs

    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()

    end = time.perf_counter()
    avg_time_ms = (end - start) / n_runs * 1000

    return avg_time_ms


# =============================================================================
# Sparsity Sweep Experiment
# =============================================================================


class SparsitySweepExperiment:
    """
    Experiment: Vary sparsity and measure accuracy.

    Sweep sparsity from 0% to target_sparsity and measure:
        - Accuracy drop
        - Model size reduction
        - Inference time
    """

    def __init__(
        self,
        sparsity_levels: Optional[List[float]] = None,
        method: PruningMethod = PruningMethod.MAGNITUDE
    ):
        """
        Initialize sparsity sweep experiment.

        Args:
            sparsity_levels: List of sparsity levels to test
            method: Pruning method to use
        """
        if sparsity_levels is None:
            sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.sparsity_levels = sparsity_levels
        self.method = method
        self.results: List[SparsityResult] = []

    def run(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        device: str = 'cuda',
        input_shape: Optional[Tuple[int, ...]] = None
    ) -> List[SparsityResult]:
        """
        Run sparsity sweep experiment.

        Args:
            model: PyTorch model (will be copied, not modified)
            dataloader: DataLoader for evaluation
            device: Device to use
            input_shape: Input shape for inference timing

        Returns:
            List of SparsityResult for each sparsity level
        """
        # Evaluate original model
        original_result = evaluate_model(model, dataloader, device)
        original_accuracy = original_result['accuracy']

        # Get original model size
        manager = PruningManager(PruningConfig(method=self.method))
        original_size_mb = manager.pruner.get_model_size_mb(model, nonzero_only=False)

        # Measure original inference time
        if input_shape is not None:
            original_time_ms = measure_inference_time(model, input_shape, device)
        else:
            original_time_ms = 0.0

        self.results = []

        for sparsity in self.sparsity_levels:
            if sparsity == 0.0:
                # Original model
                result = SparsityResult(
                    sparsity=0.0,
                    accuracy=original_accuracy,
                    accuracy_drop=0.0,
                    model_size_mb=original_size_mb,
                    compression_ratio=1.0,
                    inference_time_ms=original_time_ms
                )
            else:
                # Prune model
                pruned_model = copy.deepcopy(model)
                config = PruningConfig(method=self.method, sparsity=sparsity)
                manager = PruningManager(config)
                pruned_model = manager.prune(pruned_model, make_permanent=True)

                # Evaluate
                eval_result = evaluate_model(pruned_model, dataloader, device)
                accuracy = eval_result['accuracy']

                # Get size
                pruned_size_mb = manager.pruner.get_model_size_mb(pruned_model, nonzero_only=True)
                compression_ratio = original_size_mb / pruned_size_mb if pruned_size_mb > 0 else 1.0

                # Measure inference time
                if input_shape is not None:
                    inference_time_ms = measure_inference_time(pruned_model, input_shape, device)
                else:
                    inference_time_ms = 0.0

                result = SparsityResult(
                    sparsity=sparsity,
                    accuracy=accuracy,
                    accuracy_drop=original_accuracy - accuracy,
                    model_size_mb=pruned_size_mb,
                    compression_ratio=compression_ratio,
                    inference_time_ms=inference_time_ms
                )

            self.results.append(result)
            logger.info(f"Sparsity {sparsity:.0%}: accuracy={result.accuracy:.4f}, "
                       f"drop={result.accuracy_drop:.4f}, compression={result.compression_ratio:.2f}x")

        return self.results

    def get_report(self) -> ExperimentReport:
        """Generate experiment report."""
        results_list = [
            {
                'sparsity': r.sparsity,
                'accuracy': r.accuracy,
                'accuracy_drop': r.accuracy_drop,
                'model_size_mb': r.model_size_mb,
                'compression_ratio': r.compression_ratio,
                'inference_time_ms': r.inference_time_ms
            }
            for r in self.results
        ]

        # Find best sparsity (accuracy drop < 1%)
        best_result = None
        for r in self.results:
            if r.accuracy_drop < 0.01:
                best_result = r
                break

        summary = {
            'method': self.method.value,
            'sparsity_levels_tested': len(self.sparsity_levels),
            'best_sparsity_under_1percent_drop': best_result.sparsity if best_result else None,
            'max_compression_at_50_percent_sparsity': next(
                (r.compression_ratio for r in self.results if r.sparsity == 0.5), None
            )
        }

        return ExperimentReport(
            experiment_name="Sparsity Sweep",
            description=f"Sweep sparsity levels using {self.method.value} pruning",
            results=results_list,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# =============================================================================
# Method Comparison Experiment
# =============================================================================


class MethodComparisonExperiment:
    """
    Experiment: Compare different pruning methods.

    Compare:
        - Magnitude pruning (L1)
        - Random pruning
        - Global pruning
        - Channel pruning (structured)
    """

    METHODS = [
        PruningMethod.MAGNITUDE,
        PruningMethod.RANDOM,
        PruningMethod.GLOBAL,
        PruningMethod.CHANNEL,
    ]

    def __init(
        self,
        methods: Optional[List[PruningMethod]] = None,
        sparsity: float = 0.5
    ):
        """
        Initialize method comparison experiment.

        Args:
            methods: List of methods to compare
            sparsity: Sparsity level for comparison
        """
        self.methods = methods or self.METHODS
        self.sparsity = sparsity
        self.results: List[MethodResult] = []

    def run(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        device: str = 'cuda'
    ) -> List[MethodResult]:
        """
        Run method comparison experiment.

        Args:
            model: PyTorch model (will be copied for each method)
            dataloader: DataLoader for evaluation
            device: Device to use

        Returns:
            List of MethodResult for each method
        """
        # Evaluate original model
        original_result = evaluate_model(model, dataloader, device)
        original_accuracy = original_result['accuracy']

        # Get original model size
        manager = PruningManager()
        original_size_mb = manager.pruner.get_model_size_mb(model, nonzero_only=False)

        self.results = []

        for method in self.methods:
            # Create fresh copy
            pruned_model = copy.deepcopy(model)

            # Prune
            config = PruningConfig(method=method, sparsity=self.sparsity)
            method_manager = PruningManager(config)
            pruned_model = method_manager.prune(pruned_model, make_permanent=True)

            # Evaluate
            eval_result = evaluate_model(pruned_model, dataloader, device)
            accuracy = eval_result['accuracy']

            # Get size
            pruned_size_mb = method_manager.pruner.get_model_size_mb(pruned_model, nonzero_only=True)
            compression_ratio = original_size_mb / pruned_size_mb if pruned_size_mb > 0 else 1.0

            result = MethodResult(
                method_name=method.value,
                sparsity=self.sparsity,
                accuracy=accuracy,
                accuracy_drop=original_accuracy - accuracy,
                model_size_mb=pruned_size_mb,
                compression_ratio=compression_ratio
            )

            self.results.append(result)
            logger.info(f"Method {method.value}: accuracy={accuracy:.4f}, "
                       f"drop={result.accuracy_drop:.4f}, compression={compression_ratio:.2f}x")

        return self.results

    def get_report(self) -> ExperimentReport:
        """Generate experiment report."""
        results_list = [
            {
                'method': r.method_name,
                'accuracy': r.accuracy,
                'accuracy_drop': r.accuracy_drop,
                'model_size_mb': r.model_size_mb,
                'compression_ratio': r.compression_ratio
            }
            for r in self.results
        ]

        # Find best method
        best_method = min(self.results, key=lambda r: r.accuracy_drop)

        summary = {
            'sparsity': self.sparsity,
            'best_method': best_method.method_name,
            'best_accuracy': best_method.accuracy,
            'best_accuracy_drop': best_method.accuracy_drop,
            'methods_compared': [m.value for m in self.methods]
        }

        return ExperimentReport(
            experiment_name="Method Comparison",
            description=f"Compare pruning methods at {self.sparsity:.0%} sparsity",
            results=results_list,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# =============================================================================
# Fine-Tuning Pipeline
# =============================================================================


class FineTuningPipeline:
    """
    Fine-tuning pipeline after pruning.

    Provides:
        - Standard fine-tuning loop
        - Learning rate scheduling
        - Early stopping
        - Accuracy recovery tracking
    """

    def __init__(
        self,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = 'cosine',
        early_stopping_patience: int = 3,
        use_amp: bool = True
    ):
        """
        Initialize fine-tuning pipeline.

        Args:
            epochs: Number of fine-tuning epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            lr_scheduler: Learning rate scheduler type
            early_stopping_patience: Early stopping patience
            use_amp: Use automatic mixed precision
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        optimizer: "optim.Optimizer",
        criterion: Callable,
        device: str,
        scaler: Optional["GradScaler"] = None
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            model: PyTorch model
            dataloader: Training DataLoader
            optimizer: Optimizer
            criterion: Loss function
            device: Device
            scaler: GradScaler for AMP

        Returns:
            Tuple of (average loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if self.use_amp and scaler is not None and device.startswith('cuda'):
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def run(
        self,
        model: "nn.Module",
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        device: str = 'cuda',
        criterion: Optional[Callable] = None
    ) -> "nn.Module":
        """
        Run fine-tuning.

        Args:
            model: Pruned model to fine-tune
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to use
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Fine-tuned model
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        # Learning rate scheduler
        if self.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        elif self.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs // 3, gamma=0.1)
        else:
            scheduler = None

        # AMP scaler
        scaler = GradScaler() if self.use_amp and device.startswith('cuda') else None

        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None

        # Reset history
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        logger.info(f"Starting fine-tuning for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )

            # Validate
            val_result = evaluate_model(model, val_loader, device)
            val_acc = val_result['accuracy']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                       f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                       f"val_acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(f"Fine-tuning complete. Best val accuracy: {best_val_acc:.4f}")
        return model

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history


# =============================================================================
# Complete Pruning Experiment
# =============================================================================


class CompletePruningExperiment:
    """
    Complete pruning experiment with fine-tuning.

    Runs the full pipeline:
        1. Evaluate original model
        2. Prune model
        3. Fine-tune
        4. Compare before/after
    """

    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        fine_tune_epochs: int = 10
    ):
        """
        Initialize complete experiment.

        Args:
            config: Pruning configuration
            fine_tune_epochs: Epochs for fine-tuning
        """
        self.config = config or PruningConfig()
        self.fine_tune_epochs = fine_tune_epochs

    def run(
        self,
        model: "nn.Module",
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        test_loader: Optional["DataLoader"] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Run complete pruning experiment.

        Args:
            model: PyTorch model
            train_loader: Training data
            val_loader: Validation data
            test_loader: Test data (optional)
            device: Device to use

        Returns:
            Dictionary with experiment results
        """
        results = {}

        # 1. Evaluate original model
        logger.info("Evaluating original model...")
        original_eval = evaluate_model(model, val_loader, device)
        results['original'] = {
            'accuracy': original_eval['accuracy'],
            'sparsity': 0.0
        }

        manager = PruningManager(self.config)
        manager.save_original_model(model)
        results['original']['size_mb'] = manager.pruner.get_model_size_mb(model)

        # 2. Prune model
        logger.info(f"Pruning model to {self.config.sparsity:.0%} sparsity...")
        pruned_model = copy.deepcopy(model)
        pruned_model = manager.prune(pruned_model, make_permanent=True)

        # Evaluate pruned (before fine-tuning)
        pruned_eval = evaluate_model(pruned_model, val_loader, device)
        results['pruned_before_ft'] = {
            'accuracy': pruned_eval['accuracy'],
            'sparsity': manager.pruner.get_global_sparsity(pruned_model),
            'size_mb': manager.pruner.get_model_size_mb(pruned_model, nonzero_only=True)
        }

        # 3. Fine-tune
        logger.info(f"Fine-tuning for {self.fine_tune_epochs} epochs...")
        pipeline = FineTuningPipeline(epochs=self.fine_tune_epochs)
        pruned_model = pipeline.run(pruned_model, train_loader, val_loader, device)

        # Evaluate after fine-tuning
        ft_eval = evaluate_model(pruned_model, val_loader, device)
        results['pruned_after_ft'] = {
            'accuracy': ft_eval['accuracy'],
            'sparsity': manager.pruner.get_global_sparsity(pruned_model),
            'size_mb': manager.pruner.get_model_size_mb(pruned_model, nonzero_only=True)
        }
        results['fine_tune_history'] = pipeline.get_history()

        # 4. Test set evaluation (if provided)
        if test_loader is not None:
            test_eval = evaluate_model(pruned_model, test_loader, device)
            results['test_accuracy'] = test_eval['accuracy']

        # 5. Calculate metrics
        results['summary'] = {
            'accuracy_drop_before_ft': results['original']['accuracy'] - results['pruned_before_ft']['accuracy'],
            'accuracy_recovery': results['pruned_after_ft']['accuracy'] - results['pruned_before_ft']['accuracy'],
            'final_accuracy_drop': results['original']['accuracy'] - results['pruned_after_ft']['accuracy'],
            'compression_ratio': results['original']['size_mb'] / results['pruned_after_ft']['size_mb']
                             if results['pruned_after_ft']['size_mb'] > 0 else 1.0
        }

        logger.info(f"Experiment complete:")
        logger.info(f"  Original accuracy: {results['original']['accuracy']:.4f}")
        logger.info(f"  Pruned accuracy (before FT): {results['pruned_before_ft']['accuracy']:.4f}")
        logger.info(f"  Pruned accuracy (after FT): {results['pruned_after_ft']['accuracy']:.4f}")
        logger.info(f"  Compression ratio: {results['summary']['compression_ratio']:.2f}x")

        return results


# =============================================================================
# Run All Experiments
# =============================================================================


def run_all_experiments(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    device: str = 'cuda',
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, ExperimentReport]:
    """
    Run all pruning experiments.

    Args:
        model: PyTorch model
        train_loader: Training data
        val_loader: Validation data
        device: Device to use
        output_dir: Directory to save reports

    Returns:
        Dictionary of experiment reports
    """
    reports = {}

    # 1. Sparsity sweep
    logger.info("=" * 50)
    logger.info("Running Sparsity Sweep Experiment...")
    sweep = SparsitySweepExperiment()
    sweep.run(model, val_loader, device)
    reports['sparsity_sweep'] = sweep.get_report()

    # 2. Method comparison
    logger.info("=" * 50)
    logger.info("Running Method Comparison Experiment...")
    comparison = MethodComparisonExperiment(sparsity=0.5)
    comparison.run(model, val_loader, device)
    reports['method_comparison'] = comparison.get_report()

    # 3. Complete pruning with fine-tuning
    logger.info("=" * 50)
    logger.info("Running Complete Pruning Experiment...")
    complete = CompletePruningExperiment(
        config=PruningConfig(sparsity=0.5),
        fine_tune_epochs=5
    )
    results = complete.run(model, train_loader, val_loader, device=device)

    reports['complete_experiment'] = ExperimentReport(
        experiment_name="Complete Pruning",
        description="Full pruning pipeline with fine-tuning",
        results=[results],
        summary=results['summary'],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

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


PRUNING_EXPERIMENTS_COMPONENTS = {
    'SparsityResult': SparsityResult,
    'MethodResult': MethodResult,
    'ExperimentReport': ExperimentReport,
    'SparsitySweepExperiment': SparsitySweepExperiment,
    'MethodComparisonExperiment': MethodComparisonExperiment,
    'FineTuningPipeline': FineTuningPipeline,
    'CompletePruningExperiment': CompletePruningExperiment,
    'evaluate_model': evaluate_model,
    'measure_inference_time': measure_inference_time,
    'run_all_experiments': run_all_experiments,
}
