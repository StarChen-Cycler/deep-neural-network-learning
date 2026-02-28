"""
Model Pruning Implementation for Neural Network Compression.

This module provides:
    - MagnitudePruner: L1/L2 norm-based unstructured pruning
    - GradientPruner: Gradient-based importance pruning
    - ChannelPruner: Structured channel/filter pruning
    - GlobalPruner: Cross-layer global pruning
    - IterativePruningSchedule: Gradual pruning with fine-tuning

Theory:
    Unstructured Pruning:
        - Removes individual weights based on magnitude
        - Higher compression but requires sparse hardware support
        - L1 norm: |w|, L2 norm: w²

    Structured Pruning:
        - Removes entire channels/filters/heads
        - Hardware-friendly, direct speedup
        - Based on filter importance (L1 norm of filter weights)

    Global Pruning:
        - Prunes across all layers with global threshold
        - More accurate sparsity distribution
        - Better accuracy vs compression tradeoff

    Iterative Pruning:
        - Gradual pruning over multiple steps
        - prune -> fine-tune -> prune cycle
        - Better accuracy recovery

References:
    - Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
    - Learning both Weights and Connections: https://arxiv.org/abs/1506.02626
    - Channel Pruning: https://arxiv.org/abs/1707.06168
"""

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
import logging
import warnings

import numpy as np

# Try to import torch
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import prune
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    prune = None
    DataLoader = None
    Dataset = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class PruningMethod(Enum):
    """Supported pruning methods."""
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    GRADIENT = "gradient"
    CHANNEL = "channel"
    FILTER = "filter"
    GLOBAL = "global"


class PruningNorm(Enum):
    """Norm type for importance calculation."""
    L1 = 1
    L2 = 2


@dataclass
class PruningConfig:
    """
    Configuration for model pruning.

    Attributes:
        method: Pruning method to use
        sparsity: Target sparsity ratio (0.0 to 1.0)
        norm: Norm type for importance calculation
        dim: Dimension for structured pruning (0=output channels, 1=input channels)
        prune_bias: Whether to prune bias parameters
        iterative_steps: Number of iterative pruning steps
        fine_tune_epochs: Epochs to fine-tune after each pruning step
        layers_to_prune: List of layer name patterns to prune (None = all linear/conv)
        exclude_layers: List of layer names to exclude from pruning
    """
    method: PruningMethod = PruningMethod.MAGNITUDE
    sparsity: float = 0.5
    norm: PruningNorm = PruningNorm.L1
    dim: int = 0  # For structured pruning
    prune_bias: bool = False
    iterative_steps: int = 1
    fine_tune_epochs: int = 5
    layers_to_prune: Optional[List[str]] = None
    exclude_layers: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0, 1), got {self.sparsity}")
        if self.iterative_steps < 1:
            raise ValueError(f"iterative_steps must be >= 1, got {self.iterative_steps}")


# =============================================================================
# Base Pruner Class
# =============================================================================


class BasePruner:
    """
    Base class for all pruning methods.

    Provides common utilities for:
        - Finding prunable layers
        - Computing sparsity statistics
        - Applying and removing pruning
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruner.

        Args:
            config: Pruning configuration
        """
        if config is None:
            config = PruningConfig()
        self.config = config
        self._pruned_modules: List[str] = []

    def get_prunable_layers(
        self,
        model: "nn.Module"
    ) -> List[Tuple[str, "nn.Module", str]]:
        """
        Find all prunable layers in the model.

        Args:
            model: PyTorch model

        Returns:
            List of (name, module, param_name) tuples
        """
        prunable = []

        for name, module in model.named_modules():
            # Check if layer should be excluded
            if self.config.exclude_layers:
                if any(ex in name for ex in self.config.exclude_layers):
                    continue

            # Check if layer is in target list
            if self.config.layers_to_prune:
                if not any(pattern in name for pattern in self.config.layers_to_prune):
                    continue

            # Check for prunable parameters
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                       nn.ConvTranspose1d, nn.ConvTranspose2d,
                                       nn.ConvTranspose3d)):
                    prunable.append((name, module, 'weight'))

                    # Optionally prune bias
                    if self.config.prune_bias and hasattr(module, 'bias') and module.bias is not None:
                        prunable.append((name, module, 'bias'))

        return prunable

    def get_sparsity(self, model: "nn.Module") -> Dict[str, float]:
        """
        Compute sparsity ratio for each layer.

        Args:
            model: PyTorch model

        Returns:
            Dictionary mapping layer names to sparsity ratios
        """
        sparsity = {}

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                if hasattr(weight, 'orig'):
                    # Pruned parameter
                    mask = getattr(weight, 'mask', None)
                    if mask is not None:
                        sparsity[name] = 1.0 - mask.sum().item() / mask.numel()
                else:
                    # Check for zeros
                    sparsity[name] = (weight == 0).sum().item() / weight.numel()

        return sparsity

    def get_global_sparsity(self, model: "nn.Module") -> float:
        """
        Compute global sparsity across all prunable layers.

        Args:
            model: PyTorch model

        Returns:
            Global sparsity ratio
        """
        total_params = 0
        total_zeros = 0

        for name, module, _ in self.get_prunable_layers(model):
            weight = module.weight
            if hasattr(weight, 'orig'):
                mask = getattr(weight, 'mask', None)
                if mask is not None:
                    total_params += mask.numel()
                    total_zeros += (mask == 0).sum().item()
            else:
                total_params += weight.numel()
                total_zeros += (weight == 0).sum().item()

        return total_zeros / total_params if total_params > 0 else 0.0

    def remove_pruning(self, model: "nn.Module") -> "nn.Module":
        """
        Remove pruning reparameterization and make masks permanent.

        Args:
            model: PyTorch model with pruning applied

        Returns:
            Model with permanent pruning
        """
        for name, module in model.named_modules():
            # Skip container modules (Sequential, etc.)
            if not hasattr(module, 'weight'):
                continue

            if prune.is_pruned(module):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # Already removed or not pruned

                if self.config.prune_bias and hasattr(module, 'bias') and module.bias is not None:
                    try:
                        prune.remove(module, 'bias')
                    except ValueError:
                        pass  # Bias not pruned

        return model

    def count_parameters(self, model: "nn.Module", nonzero_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            model: PyTorch model
            nonzero_only: If True, count only nonzero parameters

        Returns:
            Number of parameters
        """
        count = 0
        for param in model.parameters():
            if nonzero_only:
                count += (param != 0).sum().item()
            else:
                count += param.numel()
        return count

    def get_model_size_mb(self, model: "nn.Module", nonzero_only: bool = True) -> float:
        """
        Get model size in megabytes.

        Args:
            model: PyTorch model
            nonzero_only: If True, count only nonzero parameters

        Returns:
            Model size in MB
        """
        param_count = self.count_parameters(model, nonzero_only)
        # Assume float32 (4 bytes per parameter)
        return param_count * 4 / (1024 * 1024)


# =============================================================================
# Magnitude Pruner (Unstructured)
# =============================================================================


class MagnitudePruner(BasePruner):
    """
    Unstructured magnitude-based pruning.

    Prunes weights with smallest absolute values (L1 norm) or
    smallest squared values (L2 norm).

    Methods:
        - L1 unstructured: Prune smallest |w|
        - L2 unstructured: Prune smallest w²
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        super().__init__(config)

    def prune_layer(
        self,
        module: "nn.Module",
        param_name: str = 'weight',
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply magnitude pruning to a single layer.

        Args:
            module: Module to prune
            param_name: Parameter name ('weight' or 'bias')
            amount: Sparsity ratio (uses config if None)

        Returns:
            Pruned module
        """
        if amount is None:
            amount = self.config.sparsity

        if self.config.norm == PruningNorm.L1:
            prune.l1_unstructured(module, param_name, amount)
        else:
            # L2 norm uses the same function but with different importance
            # For L2, we use ln_structured with n=2 but unstructured variant
            # Fall back to L1 for unstructured as PyTorch doesn't have L2 unstructured
            prune.l1_unstructured(module, param_name, amount)

        return module

    def prune_model(
        self,
        model: "nn.Module",
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply magnitude pruning to entire model.

        Args:
            model: PyTorch model
            amount: Sparsity ratio (uses config if None)

        Returns:
            Pruned model
        """
        if amount is None:
            amount = self.config.sparsity

        prunable_layers = self.get_prunable_layers(model)

        for name, module, param_name in prunable_layers:
            self.prune_layer(module, param_name, amount)
            self._pruned_modules.append(name)

        logger.info(f"Magnitude pruned {len(prunable_layers)} layers to {amount:.1%} sparsity")
        return model


# =============================================================================
# Random Pruner (Unstructured)
# =============================================================================


class RandomPruner(BasePruner):
    """
    Random unstructured pruning for baseline comparison.

    Randomly prunes weights without considering importance.
    Useful for comparing against magnitude-based methods.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        super().__init__(config)

    def prune_layer(
        self,
        module: "nn.Module",
        param_name: str = 'weight',
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply random pruning to a single layer.

        Args:
            module: Module to prune
            param_name: Parameter name
            amount: Sparsity ratio

        Returns:
            Pruned module
        """
        if amount is None:
            amount = self.config.sparsity

        prune.random_unstructured(module, param_name, amount)
        return module

    def prune_model(
        self,
        model: "nn.Module",
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply random pruning to entire model.

        Args:
            model: PyTorch model
            amount: Sparsity ratio

        Returns:
            Pruned model
        """
        if amount is None:
            amount = self.config.sparsity

        prunable_layers = self.get_prunable_layers(model)

        for name, module, param_name in prunable_layers:
            self.prune_layer(module, param_name, amount)
            self._pruned_modules.append(name)

        logger.info(f"Random pruned {len(prunable_layers)} layers to {amount:.1%} sparsity")
        return model


# =============================================================================
# Gradient-based Pruner
# =============================================================================


class GradientPruner(BasePruner):
    """
    Gradient-based importance pruning.

    Prunes weights based on gradient magnitude during training.
    Weights with consistently small gradients are less important.

    Note: Requires a forward/backward pass to compute gradients.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        super().__init__(config)
        self._gradient_saliency: Dict[str, "torch.Tensor"] = {}

    def compute_saliency(self, model: "nn.Module", dataloader: "DataLoader",
                         criterion: Callable, device: str = 'cuda',
                         num_batches: int = 10) -> None:
        """
        Compute gradient-based saliency scores.

        Args:
            model: PyTorch model
            dataloader: DataLoader for computing gradients
            criterion: Loss function
            device: Device to use
            num_batches: Number of batches to average over
        """
        model.train()
        model.to(device)

        # Initialize saliency accumulators
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self._gradient_saliency[name] = torch.zeros_like(module.weight.data)

        # Accumulate gradients
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Accumulate absolute gradients
            for name, module in model.named_modules():
                if name in self._gradient_saliency:
                    if module.weight.grad is not None:
                        self._gradient_saliency[name] += module.weight.grad.abs().data

        # Average over batches
        for name in self._gradient_saliency:
            self._gradient_saliency[name] /= num_batches

    def prune_model(
        self,
        model: "nn.Module",
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply gradient-based pruning.

        Args:
            model: PyTorch model
            amount: Sparsity ratio

        Returns:
            Pruned model
        """
        if amount is None:
            amount = self.config.sparsity

        if not self._gradient_saliency:
            raise RuntimeError("Must call compute_saliency() before pruning")

        for name, module, param_name in self.get_prunable_layers(model):
            if name in self._gradient_saliency:
                saliency = self._gradient_saliency[name]
                # Create custom mask based on saliency
                threshold = torch.quantile(saliency.flatten(), amount)
                mask = saliency > threshold

                # Apply custom pruning mask
                prune.custom_from_mask(module, param_name, mask)
                self._pruned_modules.append(name)

        logger.info(f"Gradient pruned {len(self._pruned_modules)} layers to {amount:.1%} sparsity")
        return model


# =============================================================================
# Structured Channel Pruner
# =============================================================================


class ChannelPruner(BasePruner):
    """
    Structured channel/filter pruning.

    Removes entire channels (Conv1d), filters (Conv2d/Conv3d), or
    output features (Linear) based on importance.

    This is hardware-friendly as it reduces computation directly.

    Importance metric:
        - L1 norm of filter weights
        - For Conv2d: Sum of |filter| over (C_in, H, W)
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        super().__init__(config)

    def compute_filter_importance(
        self,
        module: "nn.Module",
        dim: int = 0
    ) -> "torch.Tensor":
        """
        Compute importance score for each filter/channel.

        Args:
            module: Conv or Linear layer
            dim: Dimension to prune (0=output, 1=input)

        Returns:
            Tensor of importance scores
        """
        weight = module.weight.data

        if isinstance(module, nn.Linear):
            # For linear: (out_features, in_features)
            if dim == 0:
                # Importance of output neurons
                importance = weight.abs().sum(dim=1)
            else:
                # Importance of input features
                importance = weight.abs().sum(dim=0)

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # For conv: (out_channels, in_channels, ...)
            if dim == 0:
                # Importance of output channels (filters)
                importance = weight.abs().sum(dim=tuple(range(1, weight.dim())))
            else:
                # Importance of input channels
                importance = weight.abs().sum(dim=(0,) + tuple(range(2, weight.dim())))

        else:
            raise ValueError(f"Unsupported module type: {type(module)}")

        return importance

    def prune_layer(
        self,
        module: "nn.Module",
        amount: Optional[float] = None,
        dim: Optional[int] = None
    ) -> "nn.Module":
        """
        Apply structured channel pruning to a single layer.

        Args:
            module: Module to prune
            amount: Sparsity ratio
            dim: Dimension to prune

        Returns:
            Pruned module
        """
        if amount is None:
            amount = self.config.sparsity
        if dim is None:
            dim = self.config.dim

        # Compute importance
        importance = self.compute_filter_importance(module, dim)

        # Determine number of channels to keep
        n_channels = importance.numel()
        n_to_prune = int(n_channels * amount)
        n_to_keep = n_channels - n_to_prune

        if n_to_prune == 0:
            return module

        # Get indices of least important channels
        _, indices = torch.topk(importance, n_to_prune, largest=False)

        # Create structured mask
        if dim == 0:
            mask = torch.ones(n_channels, dtype=torch.bool, device=importance.device)
        else:
            mask = torch.ones(n_channels, dtype=torch.bool, device=importance.device)

        mask[indices] = False

        # Apply Ln structured pruning (n=1 for L1 norm)
        prune.ln_structured(module, 'weight', amount, n=self.config.norm.value, dim=dim)

        return module

    def prune_model(
        self,
        model: "nn.Module",
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply structured channel pruning to entire model.

        Args:
            model: PyTorch model
            amount: Sparsity ratio

        Returns:
            Pruned model
        """
        if amount is None:
            amount = self.config.sparsity

        prunable_layers = self.get_prunable_layers(model)

        for name, module, param_name in prunable_layers:
            # Only apply structured pruning to Conv and Linear layers
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                self.prune_layer(module, amount)
                self._pruned_modules.append(name)

        logger.info(f"Channel pruned {len(self._pruned_modules)} layers to {amount:.1%} sparsity")
        return model


# =============================================================================
# Global Pruner
# =============================================================================


class GlobalPruner(BasePruner):
    """
    Global unstructured pruning across all layers.

    Instead of applying the same sparsity to each layer, global pruning
    applies a global threshold across all parameters. This allows layers
    with less important weights to be pruned more aggressively.

    Benefits:
        - Better accuracy vs compression tradeoff
        - Automatic layer-wise sparsity distribution
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        super().__init__(config)

    def prune_model(
        self,
        model: "nn.Module",
        amount: Optional[float] = None
    ) -> "nn.Module":
        """
        Apply global unstructured pruning.

        Args:
            model: PyTorch model
            amount: Global sparsity ratio

        Returns:
            Pruned model
        """
        if amount is None:
            amount = self.config.sparsity

        # Collect all parameters to prune
        parameters_to_prune = []
        for name, module, param_name in self.get_prunable_layers(model):
            parameters_to_prune.append((module, param_name))
            self._pruned_modules.append(name)

        if not parameters_to_prune:
            logger.warning("No prunable layers found")
            return model

        # Apply global pruning
        if self.config.norm == PruningNorm.L1:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        else:
            # Global L2 not directly supported, use L1
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )

        logger.info(f"Global pruned {len(parameters_to_prune)} layers to {amount:.1%} sparsity")
        return model


# =============================================================================
# Iterative Pruning Schedule
# =============================================================================


class IterativePruningSchedule:
    """
    Gradual iterative pruning with fine-tuning.

    Implements the iterative pruning strategy:
        1. Train model to convergence
        2. Prune small amount (e.g., 20%)
        3. Fine-tune to recover accuracy
        4. Repeat until target sparsity

    Based on "Lottery Ticket Hypothesis" and
    "Learning both Weights and Connections" papers.
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.8,
        n_iterations: int = 10,
        start_iteration: int = 0,
        end_iteration: int = 10
    ):
        """
        Initialize iterative pruning schedule.

        Args:
            initial_sparsity: Starting sparsity
            final_sparsity: Target sparsity
            n_iterations: Number of pruning iterations
            start_iteration: Iteration to start pruning
            end_iteration: Iteration to end pruning
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self._current_iteration = 0

    def get_sparsity_for_iteration(self, iteration: int) -> float:
        """
        Calculate target sparsity for given iteration.

        Uses cubic interpolation schedule from "Lottery Ticket Hypothesis".

        Args:
            iteration: Current iteration

        Returns:
            Target sparsity
        """
        if iteration < self.start_iteration:
            return self.initial_sparsity
        if iteration >= self.end_iteration:
            return self.final_sparsity

        # Cubic interpolation
        progress = (iteration - self.start_iteration) / (self.end_iteration - self.start_iteration)
        sparsity = (
            self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity)
            * (1 - progress) ** 3
        )
        return sparsity

    def step(self) -> float:
        """
        Advance schedule and return current sparsity.

        Returns:
            Current target sparsity
        """
        sparsity = self.get_sparsity_for_iteration(self._current_iteration)
        self._current_iteration += 1
        return sparsity

    def reset(self) -> None:
        """Reset schedule to beginning."""
        self._current_iteration = 0


# =============================================================================
# Pruning Manager
# =============================================================================


class PruningManager:
    """
    High-level pruning manager for complete pruning workflow.

    Provides:
        - Automatic pruner selection
        - Iterative pruning with fine-tuning
        - Compression benchmarking
        - Model comparison utilities
    """

    PRUNER_CLASSES = {
        PruningMethod.MAGNITUDE: MagnitudePruner,
        PruningMethod.RANDOM: RandomPruner,
        PruningMethod.GRADIENT: GradientPruner,
        PruningMethod.CHANNEL: ChannelPruner,
        PruningMethod.FILTER: ChannelPruner,  # Same as channel
        PruningMethod.GLOBAL: GlobalPruner,
    }

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruning manager.

        Args:
            config: Pruning configuration
        """
        if config is None:
            config = PruningConfig()
        self.config = config
        self.pruner = self._create_pruner()
        self._original_model: Optional["nn.Module"] = None

    def _create_pruner(self) -> BasePruner:
        """Create pruner based on configuration."""
        pruner_class = self.PRUNER_CLASSES.get(self.config.method)
        if pruner_class is None:
            raise ValueError(f"Unknown pruning method: {self.config.method}")
        return pruner_class(self.config)

    def save_original_model(self, model: "nn.Module") -> None:
        """
        Save a copy of the original model for comparison.

        Args:
            model: Original model
        """
        self._original_model = copy.deepcopy(model)

    def prune(
        self,
        model: "nn.Module",
        make_permanent: bool = False
    ) -> "nn.Module":
        """
        Apply pruning to model.

        Args:
            model: PyTorch model
            make_permanent: Whether to remove pruning reparameterization

        Returns:
            Pruned model
        """
        # Apply pruning
        if self.config.method == PruningMethod.GRADIENT:
            raise ValueError("Gradient pruning requires compute_saliency() first")

        pruned_model = self.pruner.prune_model(model, self.config.sparsity)

        if make_permanent:
            pruned_model = self.pruner.remove_pruning(pruned_model)

        return pruned_model

    def iterative_prune(
        self,
        model: "nn.Module",
        train_fn: Callable[["nn.Module"], "nn.Module"],
        final_sparsity: Optional[float] = None,
        n_iterations: Optional[int] = None
    ) -> "nn.Module":
        """
        Apply iterative pruning with fine-tuning.

        Args:
            model: PyTorch model
            train_fn: Training function that takes model and returns fine-tuned model
            final_sparsity: Target sparsity (uses config if None)
            n_iterations: Number of iterations (uses config if None)

        Returns:
            Iteratively pruned model
        """
        if final_sparsity is None:
            final_sparsity = self.config.sparsity
        if n_iterations is None:
            n_iterations = self.config.iterative_steps

        schedule = IterativePruningSchedule(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,
            n_iterations=n_iterations
        )

        # Initial training (if model not already trained)
        logger.info("Starting iterative pruning...")

        for i in range(n_iterations):
            target_sparsity = schedule.step()
            logger.info(f"Iteration {i+1}/{n_iterations}: Target sparsity = {target_sparsity:.1%}")

            # Prune to target sparsity
            self.pruner.prune_model(model, target_sparsity)

            # Fine-tune
            if i < n_iterations - 1:  # No fine-tune after last iteration
                logger.info(f"Fine-tuning for {self.config.fine_tune_epochs} epochs...")
                model = train_fn(model)

        return model

    def get_compression_stats(
        self,
        model: "nn.Module"
    ) -> Dict[str, Any]:
        """
        Get compression statistics.

        Args:
            model: Pruned model

        Returns:
            Dictionary with compression statistics
        """
        stats = {
            'layer_sparsity': self.pruner.get_sparsity(model),
            'global_sparsity': self.pruner.get_global_sparsity(model),
            'total_params': self.pruner.count_parameters(model, nonzero_only=False),
            'nonzero_params': self.pruner.count_parameters(model, nonzero_only=True),
            'model_size_mb': self.pruner.get_model_size_mb(model, nonzero_only=True),
            'original_size_mb': None,
            'compression_ratio': None
        }

        if self._original_model is not None:
            original_size = self.pruner.get_model_size_mb(self._original_model, nonzero_only=False)
            stats['original_size_mb'] = original_size
            if stats['model_size_mb'] > 0:
                stats['compression_ratio'] = original_size / stats['model_size_mb']

        return stats

    def compare_models(
        self,
        model1: "nn.Module",
        model2: "nn.Module",
        names: Tuple[str, str] = ("Model 1", "Model 2")
    ) -> Dict[str, Any]:
        """
        Compare two models.

        Args:
            model1: First model
            model2: Second model
            names: Names for the models

        Returns:
            Comparison dictionary
        """
        return {
            names[0]: {
                'sparsity': self.pruner.get_global_sparsity(model1),
                'params': self.pruner.count_parameters(model1),
                'size_mb': self.pruner.get_model_size_mb(model1)
            },
            names[1]: {
                'sparsity': self.pruner.get_global_sparsity(model2),
                'params': self.pruner.count_parameters(model2),
                'size_mb': self.pruner.get_model_size_mb(model2)
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_pruner(
    method: str = "magnitude",
    sparsity: float = 0.5,
    **kwargs
) -> BasePruner:
    """
    Factory function to create a pruner.

    Args:
        method: Pruning method name
        sparsity: Target sparsity
        **kwargs: Additional configuration options

    Returns:
        Pruner instance
    """
    method_map = {
        'magnitude': PruningMethod.MAGNITUDE,
        'random': PruningMethod.RANDOM,
        'gradient': PruningMethod.GRADIENT,
        'channel': PruningMethod.CHANNEL,
        'filter': PruningMethod.FILTER,
        'global': PruningMethod.GLOBAL,
    }

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

    config = PruningConfig(
        method=method_map[method],
        sparsity=sparsity,
        **kwargs
    )

    pruner_class = PruningManager.PRUNER_CLASSES[config.method]
    return pruner_class(config)


def prune_model(
    model: "nn.Module",
    sparsity: float = 0.5,
    method: str = "magnitude",
    **kwargs
) -> "nn.Module":
    """
    Convenience function to prune a model.

    Args:
        model: PyTorch model
        sparsity: Target sparsity
        method: Pruning method
        **kwargs: Additional options

    Returns:
        Pruned model
    """
    manager = PruningManager(PruningConfig(
        method=PruningMethod(method),
        sparsity=sparsity,
        **kwargs
    ))
    return manager.prune(model)


def get_model_sparsity(model: "nn.Module") -> float:
    """
    Get global sparsity of a model.

    Args:
        model: PyTorch model

    Returns:
        Global sparsity ratio
    """
    pruner = BasePruner()
    return pruner.get_global_sparsity(model)


def count_zero_weights(model: "nn.Module") -> Tuple[int, int]:
    """
    Count zero and total weights in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (zero_count, total_count)
    """
    total = 0
    zeros = 0

    for param in model.parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()

    return zeros, total


# =============================================================================
# Registry
# =============================================================================


PRUNING_COMPONENTS = {
    'PruningMethod': PruningMethod,
    'PruningNorm': PruningNorm,
    'PruningConfig': PruningConfig,
    'BasePruner': BasePruner,
    'MagnitudePruner': MagnitudePruner,
    'RandomPruner': RandomPruner,
    'GradientPruner': GradientPruner,
    'ChannelPruner': ChannelPruner,
    'GlobalPruner': GlobalPruner,
    'IterativePruningSchedule': IterativePruningSchedule,
    'PruningManager': PruningManager,
    'create_pruner': create_pruner,
    'prune_model': prune_model,
    'get_model_sparsity': get_model_sparsity,
    'count_zero_weights': count_zero_weights,
}
