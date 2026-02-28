"""
Transfer Learning and Fine-tuning Implementation.

This module provides transfer learning utilities for using pretrained models:
    - TransferLearner: Wrapper for pretrained models with custom heads
    - freeze_backbone: Freeze model backbone parameters
    - unfreeze_layers: Selectively unfreeze layers
    - get_discriminative_lr_params: Layer-wise learning rate groups

Theory:
    Transfer Learning:
        Using knowledge from a model trained on a large dataset (e.g., ImageNet)
        to solve a related task with less data.

    Fine-tuning Strategies:
        1. Feature Extraction: Freeze backbone, train only the head
           - Best for: Small datasets, similar tasks
           - Learning rate: Only for new layers

        2. Partial Fine-tuning: Unfreeze some backbone layers
           - Best for: Medium datasets, related tasks
           - Learning rate: Lower for backbone, higher for head

        3. Full Fine-tuning: Train all layers
           - Best for: Large datasets, different tasks
           - Learning rate: Discriminative (lower for early layers)

    Discriminative Learning Rates:
        Earlier layers learn generic features (edges, textures)
        Later layers learn task-specific features
        Use lower LR for early layers: [1e-5, 1e-4, 1e-3, 1e-2]

References:
    - CS231n: Transfer Learning (Stanford)
    - ULMFiT: Universal Language Model Fine-tuning (Howard & Ruder, 2018)
    - torchvision models documentation
"""

from typing import List, Optional, Dict, Any, Callable, Union, Tuple
import numpy as np

# Check for PyTorch availability (required for transfer learning with pretrained models)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms
    from torchvision.models import ResNet50_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TransferLearner:
    """
    Transfer Learning wrapper for pretrained vision models.

    Provides:
        - Pretrained model loading (ResNet, etc.)
        - Custom classifier head attachment
        - Freeze/unfreeze strategies
        - Discriminative learning rate support

    Example:
        >>> learner = TransferLearner(
        ...     backbone='resnet50',
        ...     num_classes=10,
        ...     pretrained=True
        ... )
        >>> learner.freeze_backbone()
        >>> # Train only the classifier head
        >>> learner.unfreeze_backbone(num_layers=2)  # Unfreeze last 2 layers
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize TransferLearner.

        Args:
            backbone: Backbone architecture name ('resnet50', 'resnet18', etc.)
            num_classes: Number of output classes for the new task
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for the classifier head
            hidden_dim: Optional hidden dimension for two-layer head
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TransferLearner. "
                "Install with: pip install torch torchvision"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Load backbone
        self.backbone_model = self._load_backbone(backbone, pretrained)

        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()

        # Replace classifier head
        self.classifier = self._create_classifier(dropout_rate, hidden_dim)

        # Track which layers are frozen
        self._frozen_layers: List[str] = []

    def _load_backbone(self, name: str, pretrained: bool) -> nn.Module:
        """Load pretrained backbone model."""
        name_lower = name.lower()

        if name_lower == "resnet50":
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
                model = models.resnet50(weights=weights)
            else:
                model = models.resnet50(weights=None)

            # Remove the original FC layer
            self._backbone_features = nn.Sequential(*list(model.children())[:-1])
            self._original_fc_in_features = model.fc.in_features

        elif name_lower == "resnet18":
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)

            self._backbone_features = nn.Sequential(*list(model.children())[:-1])
            self._original_fc_in_features = model.fc.in_features

        elif name_lower == "resnet34":
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet34(weights=None)

            self._backbone_features = nn.Sequential(*list(model.children())[:-1])
            self._original_fc_in_features = model.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone: {name}. Use resnet18, resnet34, or resnet50.")

        return self._backbone_features

    def _get_feature_dim(self) -> int:
        """Get the output feature dimension from backbone."""
        return self._original_fc_in_features

    def _create_classifier(self, dropout_rate: float, hidden_dim: Optional[int]) -> nn.Module:
        """Create custom classifier head."""
        layers = []

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        if hidden_dim is not None:
            # Two-layer classifier
            layers.append(nn.Linear(self.feature_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_dim, self.num_classes))
        else:
            # Single-layer classifier
            layers.append(nn.Linear(self.feature_dim, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Extract features from backbone
        features = self.backbone_model(x)

        # Flatten features
        features = features.view(features.size(0), -1)

        # Classify
        logits = self.classifier(features)

        return logits

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        """Call forward pass."""
        return self.forward(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.

        Use for feature extraction: backbone acts as fixed feature extractor,
        only the classifier head is trained.
        """
        for param in self.backbone_model.parameters():
            param.requires_grad = False

        self._frozen_layers = ["all"]
        print(f"Frozen all backbone layers ({sum(p.numel() for p in self.backbone_model.parameters())} parameters)")

    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone parameters.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                       None means unfreeze all layers.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone_model.parameters():
                param.requires_grad = True
            self._frozen_layers = []
            print("Unfrozen all backbone layers")
        else:
            # Get all children of backbone
            children = list(self.backbone_model.children())

            # First freeze all
            for param in self.backbone_model.parameters():
                param.requires_grad = False

            # Then unfreeze last num_layers
            unfrozen_count = 0
            for child in reversed(children):
                if unfrozen_count >= num_layers:
                    break
                for param in child.parameters():
                    param.requires_grad = True
                unfrozen_count += 1

            self._frozen_layers = [f"last_{num_layers}_unfrozen"]
            print(f"Unfrozen last {unfrozen_count} backbone layers")

    def freeze_layer(self, layer_name: str) -> None:
        """
        Freeze a specific layer by name.

        Args:
            layer_name: Name of the layer to freeze
        """
        for name, param in self.backbone_model.named_parameters():
            if layer_name in name:
                param.requires_grad = False

        self._frozen_layers.append(layer_name)

    def get_parameter_groups(
        self,
        base_lr: float = 1e-3,
        lr_mult: float = 0.1,
        use_discriminative: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with discriminative learning rates.

        Creates parameter groups for use with PyTorch optimizers:
            - Classifier head: base_lr
            - Backbone layers: base_lr * lr_mult (or discriminative)

        Args:
            base_lr: Base learning rate for classifier head
            lr_mult: Learning rate multiplier for backbone
            use_discriminative: Use layer-wise discriminative LRs

        Returns:
            List of parameter group dictionaries

        Example:
            >>> learner = TransferLearner('resnet50', num_classes=10)
            >>> param_groups = learner.get_parameter_groups(
            ...     base_lr=1e-3,
            ...     lr_mult=0.1,
            ...     use_discriminative=True
            ... )
            >>> optimizer = torch.optim.Adam(param_groups)
        """
        if use_discriminative:
            return get_discriminative_lr_params(
                self.backbone_model,
                self.classifier,
                base_lr=base_lr,
                lr_mult=lr_mult,
            )
        else:
            # Simple two-group setup
            backbone_params = [p for p in self.backbone_model.parameters() if p.requires_grad]
            classifier_params = [p for p in self.classifier.parameters()]

            groups = []
            if backbone_params:
                groups.append({
                    "params": backbone_params,
                    "lr": base_lr * lr_mult,
                    "name": "backbone",
                })
            if classifier_params:
                groups.append({
                    "params": classifier_params,
                    "lr": base_lr,
                    "name": "classifier",
                })

            return groups

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameters(self):
        """Return all model parameters."""
        for param in self.backbone_model.parameters():
            yield param
        for param in self.classifier.parameters():
            yield param

    def named_parameters(self):
        """Return named parameters."""
        for name, param in self.backbone_model.named_parameters():
            yield f"backbone.{name}", param
        for name, param in self.classifier.named_parameters():
            yield f"classifier.{name}", param

    def train(self) -> "TransferLearner":
        """Set model to training mode."""
        self.backbone_model.train()
        self.classifier.train()
        return self

    def eval(self) -> "TransferLearner":
        """Set model to evaluation mode."""
        self.backbone_model.eval()
        self.classifier.eval()
        return self

    def to(self, device: "torch.device") -> "TransferLearner":
        """Move model to device."""
        self.backbone_model = self.backbone_model.to(device)
        self.classifier = self.classifier.to(device)
        return self

    def get_preprocess_transforms(self) -> "transforms.Compose":
        """
        Get the standard preprocessing transforms for the backbone.

        Returns:
            torchvision transforms compose object
        """
        if self.backbone_name.lower().startswith("resnet"):
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Default transforms
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters in a model's backbone.

    Args:
        model: PyTorch model

    Returns:
        Model with frozen parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_layers(model: nn.Module, num_layers: int) -> nn.Module:
    """
    Unfreeze the last N layers of a model.

    Args:
        model: PyTorch model
        num_layers: Number of layers to unfreeze from the end

    Returns:
        Model with unfrozen layers
    """
    children = list(model.children())

    # First freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze last num_layers
    for i, child in enumerate(reversed(children)):
        if i >= num_layers:
            break
        for param in child.parameters():
            param.requires_grad = True

    return model


def get_discriminative_lr_params(
    backbone: nn.Module,
    classifier: nn.Module,
    base_lr: float = 1e-3,
    lr_mult: float = 0.1,
    num_groups: int = 3,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with discriminative learning rates.

    Assigns different learning rates to different layer groups:
        - Early layers (generic features): lowest LR
        - Middle layers: medium LR
        - Later layers (task-specific): higher LR
        - Classifier head: highest LR

    This follows the "slanted triangular" learning rate schedule
    from ULMFiT, which has been shown to improve transfer learning.

    Args:
        backbone: Backbone model (pretrained)
        classifier: Classifier head (new layers)
        base_lr: Base learning rate for classifier
        lr_mult: Multiplier for backbone LR (base_lr * lr_mult)
        num_groups: Number of layer groups for discriminative LR

    Returns:
        List of parameter group dictionaries for optimizer

    Example:
        >>> param_groups = get_discriminative_lr_params(
        ...     backbone=model.backbone,
        ...     classifier=model.fc,
        ...     base_lr=1e-3,
        ...     lr_mult=0.1,
        ...     num_groups=3
        ... )
        >>> optimizer = torch.optim.Adam(param_groups)
    """
    # Get backbone layer names
    backbone_layers = list(backbone.children())
    n_layers = len(backbone_layers)

    # Calculate layer boundaries for grouping
    group_size = max(1, n_layers // num_groups)

    param_groups = []

    # Create groups for backbone layers
    for i, layer in enumerate(backbone_layers):
        # Calculate which group this layer belongs to
        group_idx = min(i // group_size, num_groups - 1)

        # Calculate LR for this group (exponential decay)
        # Later groups (higher index) get higher LR
        layer_lr = base_lr * lr_mult * (lr_mult ** (group_idx / num_groups))

        # Only include trainable parameters
        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": layer_lr,
                "name": f"backbone_group_{group_idx}",
                "layer_idx": i,
            })

    # Add classifier parameters with base LR
    classifier_params = [p for p in classifier.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({
            "params": classifier_params,
            "lr": base_lr,
            "name": "classifier",
        })

    return param_groups


def create_layerwise_lr_groups(
    model: nn.Module,
    base_lr: float = 1e-3,
    lr_decay: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Each layer gets a learning rate that decays exponentially from
    the classifier head to the first layer:

        lr_layer = base_lr * (lr_decay ^ (num_layers - layer_idx))

    Args:
        model: PyTorch model
        base_lr: Base learning rate (for classifier/final layers)
        lr_decay: Decay factor per layer

    Returns:
        List of parameter group dictionaries
    """
    layers = list(model.children())
    n_layers = len(layers)

    param_groups = []

    for i, layer in enumerate(layers):
        # Distance from end (0 = last layer)
        distance_from_end = n_layers - i - 1

        # Calculate LR with decay
        layer_lr = base_lr * (lr_decay ** distance_from_end)

        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": layer_lr,
                "name": f"layer_{i}",
            })

    return param_groups


class FineTuningStrategy:
    """
    Fine-tuning strategy manager.

    Provides pre-configured strategies for different scenarios:
        - "freeze": Freeze backbone, train classifier only
        - "partial": Unfreeze last N backbone layers
        - "full": Train all layers
        - "discriminative": Layer-wise discriminative LRs

    Example:
        >>> strategy = FineTuningStrategy("freeze")
        >>> optimizer = strategy.setup_optimizer(learner, lr=1e-3)
    """

    STRATEGIES = ["freeze", "partial", "full", "discriminative"]

    def __init__(
        self,
        strategy: str = "freeze",
        unfreeze_layers: int = 2,
        lr_mult: float = 0.1,
        discriminative_groups: int = 3,
    ):
        """
        Initialize fine-tuning strategy.

        Args:
            strategy: Strategy name ('freeze', 'partial', 'full', 'discriminative')
            unfreeze_layers: Number of layers to unfreeze for 'partial' strategy
            lr_mult: Learning rate multiplier for backbone vs classifier
            discriminative_groups: Number of groups for discriminative LR
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {self.STRATEGIES}")

        self.strategy = strategy
        self.unfreeze_layers = unfreeze_layers
        self.lr_mult = lr_mult
        self.discriminative_groups = discriminative_groups

    def apply(self, learner: TransferLearner) -> None:
        """
        Apply the fine-tuning strategy to a learner.

        Args:
            learner: TransferLearner instance
        """
        if self.strategy == "freeze":
            learner.freeze_backbone()

        elif self.strategy == "partial":
            learner.freeze_backbone()
            learner.unfreeze_backbone(num_layers=self.unfreeze_layers)

        elif self.strategy == "full":
            learner.unfreeze_backbone()  # Unfreeze all

        elif self.strategy == "discriminative":
            learner.unfreeze_backbone()  # Unfreeze all for discriminative LR

    def get_optimizer(
        self,
        learner: TransferLearner,
        lr: float = 1e-3,
        optimizer_class: type = optim.Adam,
        **optimizer_kwargs,
    ) -> "optim.Optimizer":
        """
        Create optimizer with appropriate parameter groups.

        Args:
            learner: TransferLearner instance
            lr: Base learning rate
            optimizer_class: Optimizer class (default: Adam)
            **optimizer_kwargs: Additional optimizer arguments

        Returns:
            Configured optimizer
        """
        self.apply(learner)

        if self.strategy == "discriminative":
            param_groups = learner.get_parameter_groups(
                base_lr=lr,
                lr_mult=self.lr_mult,
                use_discriminative=True,
            )
        else:
            param_groups = learner.get_parameter_groups(
                base_lr=lr,
                lr_mult=self.lr_mult,
                use_discriminative=False,
            )

        return optimizer_class(param_groups, **optimizer_kwargs)


def train_with_transfer_learning(
    learner: TransferLearner,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    strategy: str = "freeze",
    device: Optional["torch.device"] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a transfer learning model.

    Args:
        learner: TransferLearner instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        strategy: Fine-tuning strategy
        device: Device to train on
        verbose: Print training progress

    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learner = learner.to(device)
    learner.train()

    # Setup strategy and optimizer
    ft_strategy = FineTuningStrategy(strategy)
    optimizer = ft_strategy.get_optimizer(learner, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        learner.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = learner(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # Validation
        learner.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = learner(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Record history
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(100.0 * train_correct / train_total)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(100.0 * val_correct / val_total)

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {history['train_acc'][-1]:.2f}%, "
                f"Val Loss: {history['val_loss'][-1]:.4f}, "
                f"Val Acc: {history['val_acc'][-1]:.2f}%"
            )

    return history


# Convenience functions
def create_resnet50_transfer(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> TransferLearner:
    """
    Create a ResNet50 transfer learner for custom classification.

    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained ImageNet weights
        freeze_backbone: Freeze backbone for feature extraction

    Returns:
        TransferLearner instance

    Example:
        >>> learner = create_resnet50_transfer(num_classes=10, freeze_backbone=True)
        >>> optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)
    """
    learner = TransferLearner(
        backbone="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
    )

    if freeze_backbone:
        learner.freeze_backbone()

    return learner


# Registry of transfer learning utilities
TRANSFER_LEARNING_FUNCTIONS = {
    "freeze_backbone": freeze_backbone,
    "unfreeze_layers": unfreeze_layers,
    "get_discriminative_lr_params": get_discriminative_lr_params,
    "create_layerwise_lr_groups": create_layerwise_lr_groups,
    "create_resnet50_transfer": create_resnet50_transfer,
    "train_with_transfer_learning": train_with_transfer_learning,
}
