"""
Fine-tuning Strategy Comparison Experiments.

This module provides experiments comparing different fine-tuning strategies:
    - Strategy comparison on CIFAR10 with pretrained ResNet50
    - Discriminative learning rate experiments
    - Convergence speed analysis
    - Layer-wise learning rate ablation

Theory:
    Fine-tuning Strategies:
        1. Freeze: Only train classifier head
           - Fast convergence, but may underfit
           - Best for: Small datasets, similar domains

        2. Partial: Train last N backbone layers + head
           - Balance between speed and adaptation
           - Best for: Medium datasets

        3. Full: Train all layers
           - Maximum adaptation, slower convergence
           - Best for: Large datasets, different domains

        4. Discriminative LR: Layer-wise learning rates
           - Best of both worlds: fast + adaptive
           - Early layers: 1e-5 to 1e-4 (generic features)
           - Later layers: 1e-4 to 1e-3 (task-specific)
           - Classifier: 1e-3 (new task)

    Convergence Criteria:
        - Loss stabilization
        - Accuracy plateau
        - Target: <50 steps to 90% of final accuracy

References:
    - ULMFiT: Universal Language Model Fine-tuning (Howard & Ruder, 2018)
    - Ramping Up and Fast Fine-tuning (Howard & Ruder, 2018)
    - torchvision transfer learning tutorial
"""

from typing import List, Optional, Dict, Any, Tuple
import time
import json
from pathlib import Path

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms
    import torchvision.datasets as datasets
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .transfer_learning import (
    TransferLearner,
    FineTuningStrategy,
    get_discriminative_lr_params,
    create_resnet50_transfer,
)


def create_synthetic_cifar10(
    num_samples: int = 1000,
    num_classes: int = 10,
    img_size: int = 32,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Create synthetic CIFAR10-like data for testing.

    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        img_size: Image size (will be resized to 224 for ResNet)

    Returns:
        Tuple of (images, labels) tensors
    """
    # Generate random images
    images = torch.randn(num_samples, 3, img_size, img_size)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return images, labels


def create_realistic_synthetic_data(
    num_samples: int = 500,
    num_classes: int = 10,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Create more realistic synthetic data with class-specific patterns.

    This creates data that has distinguishable patterns per class,
    making the classification task learnable but not trivial.

    Args:
        num_samples: Number of samples per class
        num_classes: Number of classes

    Returns:
        Tuple of (images, labels) tensors
    """
    all_images = []
    all_labels = []

    samples_per_class = num_samples // num_classes

    for class_idx in range(num_classes):
        # Create class-specific patterns
        base_pattern = torch.zeros(3, 224, 224)

        # Different color dominance per class
        base_pattern[class_idx % 3, :, :] = 0.5 + 0.3 * (class_idx / num_classes)

        # Add spatial patterns
        if class_idx < 5:
            # Horizontal stripes
            for i in range(0, 224, 40):
                base_pattern[:, i:i+20, :] += 0.3
        else:
            # Vertical stripes
            for i in range(0, 224, 40):
                base_pattern[:, :, i:i+20] += 0.3

        # Add noise and create samples
        for _ in range(samples_per_class):
            noise = torch.randn(3, 224, 224) * 0.1
            sample = base_pattern + noise
            sample = torch.clamp(sample, -1, 1)
            all_images.append(sample)
            all_labels.append(class_idx)

    # Stack and shuffle
    images = torch.stack(all_images)
    labels = torch.tensor(all_labels)

    # Shuffle
    perm = torch.randperm(len(images))
    images = images[perm]
    labels = labels[perm]

    return images, labels


def run_finetuning_experiment(
    strategy: str,
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    num_epochs: int = 20,
    lr: float = 1e-3,
    lr_mult: float = 0.1,
    device: Optional["torch.device"] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single fine-tuning experiment.

    Args:
        strategy: Fine-tuning strategy name
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Base learning rate
        lr_mult: Backbone LR multiplier
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary with experiment results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    learner = create_resnet50_transfer(
        num_classes=10,
        pretrained=True,
        freeze_backbone=False,  # Strategy will handle freezing
    )
    learner = learner.to(device)

    # Apply strategy and get optimizer
    ft_strategy = FineTuningStrategy(
        strategy=strategy,
        lr_mult=lr_mult,
    )
    optimizer = ft_strategy.get_optimizer(learner, lr=lr)

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    best_val_acc = 0.0
    steps_to_90 = None  # Track convergence speed
    target_acc = None

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        learner.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Resize to 224x224 for ResNet
            if data.size(-1) != 224:
                data = torch.nn.functional.interpolate(
                    data, size=(224, 224), mode='bilinear', align_corners=False
                )

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

                if data.size(-1) != 224:
                    data = torch.nn.functional.interpolate(
                        data, size=(224, 224), mode='bilinear', align_corners=False
                    )

                output = learner(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        epoch_time = time.time() - epoch_start

        # Record history
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Track convergence (steps to 90% of final accuracy)
        if target_acc is None:
            # Estimate target from early epochs
            if epoch >= 5:
                target_acc = best_val_acc * 0.9

        if steps_to_90 is None and target_acc is not None and val_acc >= target_acc:
            steps_to_90 = (epoch + 1) * len(train_loader)

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {history['val_loss'][-1]:.4f}, "
                f"Val Acc: {val_acc:.2f}%, "
                f"Time: {epoch_time:.2f}s"
            )

    total_time = time.time() - start_time

    return {
        "strategy": strategy,
        "lr": lr,
        "lr_mult": lr_mult,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "steps_to_90_percent": steps_to_90,
        "total_time": total_time,
        "history": history,
    }


def compare_finetuning_strategies(
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    strategies: Optional[List[str]] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: Optional["torch.device"] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different fine-tuning strategies.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        strategies: List of strategies to compare
        num_epochs: Number of epochs per strategy
        lr: Base learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary mapping strategy names to results
    """
    if strategies is None:
        strategies = ["freeze", "partial", "full", "discriminative"]

    results = {}

    for strategy in strategies:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running strategy: {strategy}")
            print(f"{'='*60}")

        result = run_finetuning_experiment(
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )
        results[strategy] = result

        if verbose:
            print(f"\nResults for {strategy}:")
            print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")
            print(f"  Final Val Acc: {result['final_val_acc']:.2f}%")
            print(f"  Total Time: {result['total_time']:.2f}s")

    return results


def run_discriminative_lr_ablation(
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    lr_mult_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
    num_epochs: int = 15,
    lr: float = 1e-3,
    device: Optional["torch.device"] = None,
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """
    Ablation study on discriminative learning rate multiplier.

    Tests different backbone LR multipliers to find optimal ratio
    between backbone and classifier learning rates.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        lr_mult_values: List of LR multiplier values to test
        num_epochs: Number of epochs per experiment
        lr: Base learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary mapping lr_mult values to results
    """
    results = {}

    for lr_mult in lr_mult_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing lr_mult = {lr_mult}")
            print(f"{'='*60}")

        result = run_finetuning_experiment(
            strategy="discriminative",
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            lr_mult=lr_mult,
            device=device,
            verbose=verbose,
        )
        results[lr_mult] = result

        if verbose:
            print(f"\nResults for lr_mult={lr_mult}:")
            print(f"  Best Val Acc: {result['best_val_acc']:.2f}%")

    return results


def run_convergence_analysis(
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: Optional["torch.device"] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze convergence speed of fine-tuning.

    Measures:
        - Steps to reach 90% of final accuracy
        - Loss stabilization epoch
        - Training curves

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary with convergence analysis
    """
    result = run_finetuning_experiment(
        strategy="freeze",
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    history = result["history"]
    val_accs = history["val_acc"]

    # Find convergence metrics
    final_acc = val_accs[-1]
    target_90 = final_acc * 0.9
    target_95 = final_acc * 0.95

    epoch_to_90 = None
    epoch_to_95 = None

    for i, acc in enumerate(val_accs):
        if epoch_to_90 is None and acc >= target_90:
            epoch_to_90 = i + 1
        if epoch_to_95 is None and acc >= target_95:
            epoch_to_95 = i + 1

    # Find loss stabilization (variance < threshold)
    losses = history["val_loss"]
    stabilization_epoch = None
    window = 5
    threshold = 0.01

    for i in range(window, len(losses)):
        recent_losses = losses[i-window:i]
        variance = sum((l - sum(recent_losses)/window)**2 for l in recent_losses) / window
        if variance < threshold and stabilization_epoch is None:
            stabilization_epoch = i

    result["convergence_analysis"] = {
        "final_accuracy": final_acc,
        "epoch_to_90_percent": epoch_to_90,
        "epoch_to_95_percent": epoch_to_95,
        "steps_to_90_percent": epoch_to_90 * len(train_loader) if epoch_to_90 else None,
        "loss_stabilization_epoch": stabilization_epoch,
    }

    if verbose:
        print("\nConvergence Analysis:")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Epoch to 90%: {epoch_to_90}")
        print(f"  Steps to 90%: {epoch_to_90 * len(train_loader) if epoch_to_90 else 'N/A'}")
        print(f"  Loss Stabilization: Epoch {stabilization_epoch}")

    return result


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a markdown comparison report.

    Args:
        results: Results from compare_finetuning_strategies
        output_path: Optional path to save report

    Returns:
        Markdown formatted report string
    """
    report = []
    report.append("# Fine-tuning Strategy Comparison Report\n")
    report.append("## Summary\n")

    # Summary table
    report.append("| Strategy | Best Val Acc | Final Val Acc | Total Time | Steps to 90% |")
    report.append("|----------|--------------|---------------|------------|--------------|")

    for strategy, result in results.items():
        steps = result.get("steps_to_90_percent", "N/A")
        if steps is not None:
            steps = str(steps)
        report.append(
            f"| {strategy} | {result['best_val_acc']:.2f}% | "
            f"{result['final_val_acc']:.2f}% | {result['total_time']:.2f}s | {steps} |"
        )

    report.append("\n## Detailed Results\n")

    for strategy, result in results.items():
        report.append(f"\n### {strategy.upper()}\n")
        report.append(f"- Learning Rate: {result['lr']}")
        report.append(f"- LR Multiplier: {result['lr_mult']}")
        report.append(f"- Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        report.append(f"- Final Validation Accuracy: {result['final_val_acc']:.2f}%")
        report.append(f"- Final Training Accuracy: {result['final_train_acc']:.2f}%")
        report.append(f"- Total Training Time: {result['total_time']:.2f}s")

        # Add convergence analysis if available
        if "convergence_analysis" in result:
            conv = result["convergence_analysis"]
            report.append(f"- Epoch to 90% Accuracy: {conv['epoch_to_90_percent']}")
            report.append(f"- Steps to 90% Accuracy: {conv['steps_to_90_percent']}")

    report_text = "\n".join(report)

    if output_path:
        Path(output_path).write_text(report_text)

    return report_text


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save experiment results to JSON file."""
    # Convert any non-serializable types
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (list, tuple)):
                    serializable_results[key][k] = list(v)
                elif isinstance(v, (int, float, str, bool, type(None))):
                    serializable_results[key][k] = v
                else:
                    serializable_results[key][k] = str(v)
        else:
            serializable_results[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)


def run_full_comparison_experiment(
    output_dir: str = "./results/finetuning",
    num_train_samples: int = 2000,
    num_val_samples: int = 500,
    num_epochs: int = 20,
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """
    Run complete fine-tuning comparison experiment.

    This function:
        1. Creates synthetic CIFAR10-like data
        2. Compares all fine-tuning strategies
        3. Runs discriminative LR ablation
        4. Analyzes convergence
        5. Saves results and report

    Args:
        output_dir: Directory to save results
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_epochs: Number of epochs per experiment
        device: Device to use

    Returns:
        Dictionary with all experiment results
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for fine-tuning experiments")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating synthetic data...")
    train_images, train_labels = create_realistic_synthetic_data(num_train_samples)
    val_images, val_labels = create_realistic_synthetic_data(num_val_samples)

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Run strategy comparison
    print("\n" + "="*60)
    print("Comparing fine-tuning strategies...")
    print("="*60)

    strategy_results = compare_finetuning_strategies(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
    )

    # Run discriminative LR ablation
    print("\n" + "="*60)
    print("Running discriminative LR ablation...")
    print("="*60)

    lr_ablation_results = run_discriminative_lr_ablation(
        train_loader=train_loader,
        val_loader=val_loader,
        lr_mult_values=[0.01, 0.05, 0.1, 0.2],
        num_epochs=num_epochs // 2,
        device=device,
        verbose=True,
    )

    # Run convergence analysis
    print("\n" + "="*60)
    print("Running convergence analysis...")
    print("="*60)

    convergence_result = run_convergence_analysis(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
    )

    # Generate report
    print("\nGenerating report...")
    report = generate_comparison_report(
        strategy_results,
        output_path=str(output_path / "comparison_report.md"),
    )

    # Save all results
    all_results = {
        "strategy_comparison": strategy_results,
        "lr_ablation": lr_ablation_results,
        "convergence_analysis": convergence_result,
    }

    save_results(all_results, str(output_path / "experiment_results.json"))

    print(f"\nResults saved to {output_path}")
    print("\n" + report)

    return all_results


# Convenience function for quick testing
def quick_test_transfer_learning() -> bool:
    """
    Quick test of transfer learning functionality.

    Returns:
        True if all tests pass
    """
    if not HAS_TORCH:
        print("PyTorch not available, skipping test")
        return False

    print("Testing TransferLearner creation...")
    learner = create_resnet50_transfer(num_classes=10, freeze_backbone=True)

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = learner(x)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    # Test freeze/unfreeze
    learner.freeze_backbone()
    trainable = sum(p.numel() for p in learner.parameters() if p.requires_grad)
    print(f"Trainable params after freeze: {trainable}")

    learner.unfreeze_backbone(num_layers=2)
    trainable = sum(p.numel() for p in learner.parameters() if p.requires_grad)
    print(f"Trainable params after partial unfreeze: {trainable}")

    # Test parameter groups
    groups = learner.get_parameter_groups(base_lr=1e-3, lr_mult=0.1)
    print(f"Parameter groups: {len(groups)}")
    for g in groups:
        print(f"  {g['name']}: lr={g['lr']}")

    # Test FineTuningStrategy
    ft = FineTuningStrategy("freeze")
    optimizer = ft.get_optimizer(learner, lr=1e-3)
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    quick_test_transfer_learning()
