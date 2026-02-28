"""
Knowledge Distillation Implementation for Model Compression.

This module provides:
    - KnowledgeDistiller: Teacher-student training with temperature scaling
    - DistillationLoss: Combined CE + KL divergence loss
    - FeatureDistillation: Intermediate layer matching
    - TemperatureScheduler: Dynamic temperature adjustment

Theory:
    Knowledge distillation transfers knowledge from a large "teacher" model
    to a smaller "student" model. The key insight is that soft targets
    (probability distributions over classes) contain more information than
    hard labels.

    Temperature Scaling:
        Soft labels are produced by raising logits to power 1/T:
        soft_targets = softmax(logits / T)

        Higher T produces softer distributions, revealing more information
        about which classes the teacher considers similar.

    Distillation Loss:
        L_total = α * L_CE(y, y_student) + (1-α) * T² * L_KL(p_teacher, p_student)

        where:
        - L_CE: Cross-entropy with hard labels
        - L_KL: KL divergence between soft distributions
        - T²: Scaling factor to match gradient magnitudes
        - α: Weight balancing hard vs soft targets

    Feature Distillation:
        Match intermediate layer representations:
        L_feature = ||f_teacher - f_student||²

References:
    - Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
    - FitNets: Hints for Thin Deep Nets (Romero et al., 2015)
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import os
import copy
import warnings
import math

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

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class DistillationType(Enum):
    """Types of knowledge distillation."""
    LOGIT = "logit"  # Soft target distillation (Hinton et al.)
    FEATURE = "feature"  # Feature-based distillation (FitNets)
    ATTENTION = "attention"  # Attention transfer
    COMBINED = "combined"  # Multiple distillation types


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies."""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    COSINE_DECAY = "cosine_decay"
    WARMUP_DECAY = "warmup_decay"


@dataclass
class DistillationConfig:
    """
    Configuration for knowledge distillation.

    Attributes:
        temperature: Softmax temperature (higher = softer distributions)
        alpha: Weight for hard label loss (1-alpha for soft label loss)
        distillation_type: Type of distillation to use
        temperature_schedule: Temperature scheduling strategy
        temperature_start: Starting temperature for schedules
        temperature_end: Ending temperature for schedules
        warmup_epochs: Epochs for temperature warmup
        feature_layers: Layer names for feature distillation
        feature_weight: Weight for feature distillation loss
        ce_weight: Weight for cross-entropy loss
        kl_weight: Weight for KL divergence loss
    """
    temperature: float = 4.0
    alpha: float = 0.3  # Weight for CE loss (1-alpha for KL loss)
    distillation_type: DistillationType = DistillationType.LOGIT
    temperature_schedule: TemperatureSchedule = TemperatureSchedule.CONSTANT
    temperature_start: float = 10.0
    temperature_end: float = 2.0
    warmup_epochs: int = 5
    feature_layers: Optional[List[str]] = None
    feature_weight: float = 0.1
    ce_weight: float = 0.3
    kl_weight: float = 0.7

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """
        Get temperature for current epoch based on schedule.

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs

        Returns:
            Temperature value for this epoch
        """
        if self.temperature_schedule == TemperatureSchedule.CONSTANT:
            return self.temperature

        progress = epoch / max(total_epochs - 1, 1)

        if self.temperature_schedule == TemperatureSchedule.LINEAR_DECAY:
            return self.temperature_start - (self.temperature_start - self.temperature_end) * progress

        elif self.temperature_schedule == TemperatureSchedule.COSINE_DECAY:
            return self.temperature_end + 0.5 * (self.temperature_start - self.temperature_end) * \
                   (1 + math.cos(math.pi * progress))

        elif self.temperature_schedule == TemperatureSchedule.WARMUP_DECAY:
            if epoch < self.warmup_epochs:
                return self.temperature_start + (self.temperature - self.temperature_start) * \
                       (epoch / self.warmup_epochs)
            else:
                post_warmup_progress = (epoch - self.warmup_epochs) / max(total_epochs - self.warmup_epochs - 1, 1)
                return self.temperature - (self.temperature - self.temperature_end) * post_warmup_progress

        return self.temperature


# =============================================================================
# Distillation Loss Functions
# =============================================================================


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Combines:
    - Cross-entropy loss with hard labels
    - KL divergence loss with soft targets
    - Optional feature matching loss

    Example:
        >>> loss_fn = DistillationLoss(temperature=4.0, alpha=0.3)
        >>> loss = loss_fn(student_logits, teacher_logits, targets)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Softmax temperature for soft targets
            alpha: Weight for CE loss (1-alpha for KL loss)
            label_smoothing: Label smoothing for CE loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        targets: Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            targets: Ground truth labels
            temperature: Override temperature (uses config if None)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if temperature is None:
            temperature = self.temperature

        # Cross-entropy loss with hard labels
        ce_loss = self.ce_loss(student_logits, targets)

        # KL divergence loss with soft targets
        # Scale by T² to maintain gradient magnitude
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = self.kl_loss(soft_student, soft_targets) * (temperature ** 2)

        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        loss_components = {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_components


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss.

    Matches intermediate representations between teacher and student
    using L2 distance or cosine similarity.

    Example:
        >>> loss_fn = FeatureDistillationLoss()
        >>> loss = loss_fn(student_features, teacher_features)
    """

    def __init__(
        self,
        loss_type: str = "l2",
        normalize: bool = True,
    ):
        """
        Initialize feature distillation loss.

        Args:
            loss_type: "l2" for MSE, "l1" for MAE, "cosine" for cosine similarity
            normalize: Whether to normalize features before computing loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

        if loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tensor:
        """
        Compute feature distillation loss.

        Args:
            student_features: Features from student model
            teacher_features: Features from teacher model

        Returns:
            Feature matching loss
        """
        # Normalize if requested
        if self.normalize:
            student_features = F.normalize(student_features, dim=-1)
            teacher_features = F.normalize(teacher_features, dim=-1)

        if self.loss_type == "cosine":
            # CosineEmbeddingLoss expects target of 1 (similar)
            target = torch.ones(student_features.size(0), device=student_features.device)
            return self.loss_fn(student_features, teacher_features, target)
        else:
            return self.loss_fn(student_features, teacher_features)


# =============================================================================
# Feature Extraction
# =============================================================================


class FeatureExtractor:
    """
    Extract intermediate features from neural network layers.

    Example:
        >>> extractor = FeatureExtractor(model, ["layer1", "layer2"])
        >>> features = extractor(input_tensor)
        >>> print(features["layer1"].shape)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
    ):
        """
        Initialize feature extractor.

        Args:
            model: PyTorch model to extract features from
            layer_names: Names of layers to extract features from
        """
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, Tensor] = {}
        self._hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """Create a forward hook function."""
        def hook(module, input, output):
            self.features[name] = output.detach() if isinstance(output, Tensor) else output[0].detach()
        return hook

    def __call__(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Extract features from input.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        self.features = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


# =============================================================================
# Knowledge Distiller
# =============================================================================


class KnowledgeDistiller:
    """
    Knowledge distillation trainer for compressing neural networks.

    This class implements:
    - Temperature-scaled soft target distillation
    - Feature-based distillation
    - Combined distillation strategies
    - Temperature scheduling

    Example:
        >>> distiller = KnowledgeDistiller(teacher_model, student_model, config)
        >>> distiller.train(train_loader, num_epochs=10)
        >>> student_model = distiller.get_student_model()
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize knowledge distiller.

        Args:
            teacher_model: Pre-trained teacher model (frozen)
            student_model: Student model to train
            config: Distillation configuration
            device: Device to use (auto-detected if None)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for knowledge distillation")

        self.config = config or DistillationConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup models
        self.teacher_model = teacher_model.to(self.device)
        self.student_model = student_model.to(self.device)

        # Freeze teacher
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Setup losses
        self.distillation_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
        )

        self.feature_loss = None
        self.teacher_feature_extractor = None
        self.student_feature_extractor = None

        if self.config.distillation_type in [DistillationType.FEATURE, DistillationType.COMBINED]:
            if self.config.feature_layers:
                self.feature_loss = FeatureDistillationLoss()
                self.teacher_feature_extractor = FeatureExtractor(
                    self.teacher_model, self.config.feature_layers
                )
                self.student_feature_extractor = FeatureExtractor(
                    self.student_model, self.config.feature_layers
                )

        # Training state
        self.training_history: List[Dict[str, float]] = []
        self.best_loss = float("inf")
        self.best_model_state = None

    def train(
        self,
        train_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        num_epochs: int = 10,
        lr: float = 0.001,
        scheduler: Optional[Any] = None,
        val_loader: Optional[DataLoader] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Train student model with knowledge distillation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer (created if None)
            num_epochs: Number of training epochs
            lr: Learning rate (used if optimizer is None)
            scheduler: Learning rate scheduler
            val_loader: Validation data loader
            callback: Optional callback for epoch end

        Returns:
            Training history and metrics
        """
        # Setup optimizer if not provided
        if optimizer is None:
            optimizer = optim.Adam(self.student_model.parameters(), lr=lr)

        # Training loop
        self.training_history = []

        for epoch in range(num_epochs):
            # Get temperature for this epoch
            temperature = self.config.get_temperature(epoch, num_epochs)

            # Training phase
            train_metrics = self._train_epoch(
                train_loader, optimizer, temperature, epoch
            )

            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader, temperature)

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch,
                "temperature": temperature,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.training_history.append(epoch_metrics)

            # Update best model
            total_loss = train_metrics.get("total_loss", float("inf"))
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_model_state = copy.deepcopy(self.student_model.state_dict())

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Callback
            if callback is not None:
                callback(epoch, epoch_metrics)

            # Logging
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics.get('total_loss', 0):.4f} - "
                f"Temp: {temperature:.2f}"
            )

        return {
            "history": self.training_history,
            "best_loss": self.best_loss,
            "final_train_loss": train_metrics.get("total_loss", 0),
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        temperature: float,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_feature = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)

            student_logits = self.student_model(inputs)

            # Compute distillation loss
            loss, loss_components = self.distillation_loss(
                student_logits, teacher_logits, targets, temperature
            )

            # Add feature distillation loss if enabled
            if self.feature_loss is not None and self.config.feature_layers:
                teacher_features = self.teacher_feature_extractor(inputs)
                student_features = self.student_feature_extractor(inputs)

                feature_loss = 0.0
                for layer_name in self.config.feature_layers:
                    if layer_name in teacher_features and layer_name in student_features:
                        feature_loss += self.feature_loss(
                            student_features[layer_name],
                            teacher_features[layer_name]
                        )

                loss += self.config.feature_weight * feature_loss
                total_feature += feature_loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_ce += loss_components["ce_loss"]
            total_kl += loss_components["kl_loss"]
            num_batches += 1

        return {
            "total_loss": total_loss / num_batches,
            "ce_loss": total_ce / num_batches,
            "kl_loss": total_kl / num_batches,
            "feature_loss": total_feature / num_batches if total_feature > 0 else 0,
        }

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        temperature: float,
    ) -> Dict[str, float]:
        """Validate on validation set."""
        self.student_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            teacher_logits = self.teacher_model(inputs)
            student_logits = self.student_model(inputs)

            loss, _ = self.distillation_loss(
                student_logits, teacher_logits, targets, temperature
            )

            total_loss += loss.item()
            predictions = student_logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_correct / total_samples,
        }

    def get_student_model(self) -> nn.Module:
        """Get the trained student model."""
        return self.student_model

    def load_best_model(self) -> None:
        """Load the best model state."""
        if self.best_model_state is not None:
            self.student_model.load_state_dict(self.best_model_state)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate student model performance.

        Args:
            test_loader: Test data loader

        Returns:
            Evaluation metrics
        """
        self.student_model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0

        ce_loss = nn.CrossEntropyLoss()

        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            student_logits = self.student_model(inputs)
            loss = ce_loss(student_logits, targets)

            predictions = student_logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()
            num_batches += 1

        return {
            "accuracy": total_correct / total_samples,
            "loss": total_loss / num_batches,
            "num_samples": total_samples,
        }

    def compare_inference_speed(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Compare inference speed between teacher and student.

        Args:
            input_shape: Shape of input tensor (without batch dimension)
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Speed comparison metrics
        """
        self.teacher_model.eval()
        self.student_model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape, device=self.device)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.teacher_model(dummy_input)
                _ = self.student_model(dummy_input)

        # Benchmark teacher
        teacher_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.teacher_model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            teacher_times.append((end - start) * 1000)

        # Benchmark student
        student_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.student_model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            student_times.append((end - start) * 1000)

        teacher_mean = np.mean(teacher_times)
        student_mean = np.mean(student_times)

        return {
            "teacher_latency_ms": teacher_mean,
            "student_latency_ms": student_mean,
            "speedup": teacher_mean / student_mean,
            "teacher_params": sum(p.numel() for p in self.teacher_model.parameters()),
            "student_params": sum(p.numel() for p in self.student_model.parameters()),
            "compression_ratio": sum(p.numel() for p in self.teacher_model.parameters()) /
                                max(sum(p.numel() for p in self.student_model.parameters()), 1),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_distiller(
    teacher_model: nn.Module,
    student_model: nn.Module,
    temperature: float = 4.0,
    alpha: float = 0.3,
    device: Optional[torch.device] = None,
) -> KnowledgeDistiller:
    """
    Create a knowledge distiller with default configuration.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        temperature: Softmax temperature
        alpha: Weight for hard label loss
        device: Device to use

    Returns:
        Configured KnowledgeDistiller instance
    """
    config = DistillationConfig(temperature=temperature, alpha=alpha)
    return KnowledgeDistiller(teacher_model, student_model, config, device)


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    temperature: float = 4.0,
    alpha: float = 0.3,
    lr: float = 0.001,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quick distillation function.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        train_loader: Training data loader
        num_epochs: Number of training epochs
        temperature: Softmax temperature
        alpha: Weight for hard label loss
        lr: Learning rate

    Returns:
        Tuple of (trained_student_model, training_history)
    """
    distiller = create_distiller(teacher_model, student_model, temperature, alpha)
    history = distiller.train(train_loader, num_epochs=num_epochs, lr=lr)
    return distiller.get_student_model(), history


def search_temperature(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    temperatures: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0],
    num_epochs: int = 5,
) -> Dict[str, Any]:
    """
    Search for optimal temperature.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        temperatures: List of temperatures to try
        num_epochs: Number of epochs per temperature

    Returns:
        Dictionary with temperature search results
    """
    results = {}

    for temp in temperatures:
        # Reset student model
        student_model.reset_parameters() if hasattr(student_model, 'reset_parameters') else None

        config = DistillationConfig(temperature=temp)
        distiller = KnowledgeDistiller(teacher_model, student_model, config)

        history = distiller.train(
            train_loader,
            num_epochs=num_epochs,
            val_loader=val_loader
        )

        # Evaluate final model
        metrics = distiller.evaluate(val_loader)

        results[temp] = {
            "val_accuracy": metrics["accuracy"],
            "val_loss": metrics["loss"],
            "final_train_loss": history["final_train_loss"],
        }

        logger.info(f"Temperature {temp}: Val Accuracy = {metrics['accuracy']:.4f}")

    # Find best temperature
    best_temp = max(results.keys(), key=lambda t: results[t]["val_accuracy"])

    return {
        "results": results,
        "best_temperature": best_temp,
        "best_accuracy": results[best_temp]["val_accuracy"],
    }


# =============================================================================
# Registry
# =============================================================================


DISTILLATION_COMPONENTS = {
    "enums": ["DistillationType", "TemperatureSchedule"],
    "config": ["DistillationConfig"],
    "losses": ["DistillationLoss", "FeatureDistillationLoss"],
    "extractors": ["FeatureExtractor"],
    "distiller": ["KnowledgeDistiller"],
    "functions": ["create_distiller", "distill_model", "search_temperature"],
}
