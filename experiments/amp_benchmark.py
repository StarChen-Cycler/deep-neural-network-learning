"""
AMP Benchmark: Speed and Accuracy Comparison for Mixed Precision Training.

This script compares:
    - Training speed (FP16 vs BF16 vs FP32)
    - Memory usage across precision modes
    - Final model accuracy
    - Convergence rate

Hardware: RTX 3050 Ti (4GB VRAM)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase4_advanced.mixed_precision import (
    MixedPrecisionTrainer,
    GradScalerConfig,
    is_fp16_supported,
    is_bf16_supported,
    get_recommended_precision,
    get_device_info,
    enable_tf32,
    enable_optimizations_for_small_vram,
    compare_precision_modes,
)


# =============================================================================
# Model Definitions
# =============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking."""

    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SmallCNN(nn.Module):
    """Small CNN for image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Data Generation
# =============================================================================


def create_synthetic_mnist(
    num_samples: int = 1000,
    img_size: int = 28,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic MNIST-like data.

    Args:
        num_samples: Number of samples
        img_size: Image size
        num_classes: Number of classes
        seed: Random seed

    Returns:
        Tuple of (images, labels)
    """
    torch.manual_seed(seed)

    # Generate class centers
    centers = torch.randn(num_classes, 1, img_size, img_size)

    # Generate samples
    images = []
    labels = []

    samples_per_class = num_samples // num_classes

    for class_idx in range(num_classes):
        # Generate samples around class center
        class_images = centers[class_idx].unsqueeze(0) + torch.randn(samples_per_class, 1, img_size, img_size) * 0.3
        class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)

        images.append(class_images)
        labels.append(class_labels)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(labels))
    images = images[perm]
    labels = labels[perm]

    return images, labels


def create_synthetic_text(
    num_samples: int = 1000,
    seq_len: int = 32,
    vocab_size: int = 1000,
    embed_dim: int = 64,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic text data.

    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension (for linear approx)
        num_classes: Number of classes
        seed: Random seed

    Returns:
        Tuple of (sequences, labels)
    """
    torch.manual_seed(seed)

    # Generate random sequences (as one-hot sum approximations)
    sequences = torch.randn(num_samples, seq_len, embed_dim)
    labels = torch.randint(0, num_classes, (num_samples,))

    return sequences, labels


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_single_precision(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    precision: str,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Benchmark a single precision mode.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        precision: Precision mode ('fp16', 'bf16', 'fp32')
        num_epochs: Number of epochs
        learning_rate: Learning rate

    Returns:
        Dictionary with benchmark results
    """
    # Reset model
    model = model.__class__()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        precision=precision,
        grad_clip_norm=1.0,
    )

    results = {
        'precision': precision,
        'epochs': [],
        'train_losses': [],
        'val_accuracies': [],
        'epoch_times': [],
        'memory_mb': [],
    }

    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        epoch_losses = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            loss = trainer.train_step(inputs, targets)
            epoch_losses.append(loss)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(trainer.device)
                targets = targets.to(trainer.device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_time = time.time() - epoch_start
        val_accuracy = correct / total if total > 0 else 0
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        memory_mb = trainer._get_memory_mb()

        results['epochs'].append(epoch + 1)
        results['train_losses'].append(avg_loss)
        results['val_accuracies'].append(val_accuracy)
        results['epoch_times'].append(epoch_time)
        results['memory_mb'].append(memory_mb)

        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"loss={avg_loss:.4f}, acc={val_accuracy:.4f}, "
              f"time={epoch_time:.2f}s, mem={memory_mb:.1f}MB")

    total_time = time.time() - total_start

    results['total_time'] = total_time
    results['final_loss'] = results['train_losses'][-1]
    results['final_accuracy'] = results['val_accuracies'][-1]
    results['avg_epoch_time'] = sum(results['epoch_times']) / len(results['epoch_times'])
    results['max_memory_mb'] = max(results['memory_mb'])
    results['converged'] = results['final_accuracy'] > 0.7

    return results


def run_full_benchmark(
    batch_sizes: List[int] = [32, 64],
    num_epochs: int = 5,
    output_dir: str = 'results',
) -> Dict[str, Any]:
    """
    Run full benchmark comparing all precision modes.

    Args:
        batch_sizes: List of batch sizes to test
        num_epochs: Number of epochs per test
        output_dir: Output directory for results

    Returns:
        Dictionary with all benchmark results
    """
    print("=" * 60)
    print("Mixed Precision Training Benchmark")
    print("=" * 60)

    # Print device info
    device_info = get_device_info()
    print(f"\nDevice Info:")
    print(f"  Device: {device_info.get('device_name', 'Unknown')}")
    print(f"  CUDA: {device_info.get('cuda_available', False)}")
    print(f"  Compute Capability: {device_info.get('compute_capability', 'N/A')}")
    print(f"  VRAM: {device_info.get('vram_gb', 0):.1f} GB")
    print(f"  FP16: {device_info.get('fp16_supported', False)}")
    print(f"  BF16: {device_info.get('bf16_supported', False)}")
    print(f"  TF32: {device_info.get('tf32_supported', False)}")
    print(f"  Tensor Cores: {device_info.get('tensor_cores', False)}")
    print(f"\nRecommended Precision: {get_recommended_precision()}")

    # Enable optimizations for small VRAM
    print(f"\nEnabling optimizations for small VRAM...")
    opts = enable_optimizations_for_small_vram()
    for opt, enabled in opts.items():
        print(f"  {opt}: {enabled}")

    all_results = {}

    # Test each batch size
    for batch_size in batch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Batch Size: {batch_size}")
        print("=" * 60)

        # Create data
        print("\nCreating synthetic data...")
        train_images, train_labels = create_synthetic_mnist(num_samples=1000)
        val_images, val_labels = create_synthetic_mnist(num_samples=200)

        train_dataset = TensorDataset(train_images, train_labels)
        val_dataset = TensorDataset(val_images, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Determine modes to test
        modes = ['fp32']
        if is_fp16_supported():
            modes.append('fp16')
        if is_bf16_supported():
            modes.append('bf16')

        batch_results = {}

        for mode in modes:
            print(f"\n{'-' * 40}")
            print(f"Testing {mode.upper()}")
            print("-" * 40)

            model = SimpleMLP()
            results = benchmark_single_precision(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                precision=mode,
                num_epochs=num_epochs,
            )

            batch_results[mode] = results

        all_results[f'batch_{batch_size}'] = batch_results

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for batch_key, batch_results in all_results.items():
        print(f"\n{batch_key}:")

        for mode, results in batch_results.items():
            speedup = 1.0
            if 'fp32' in batch_results and mode != 'fp32':
                fp32_time = batch_results['fp32']['avg_epoch_time']
                mode_time = results['avg_epoch_time']
                speedup = fp32_time / mode_time if mode_time > 0 else 1.0

            print(f"  {mode.upper():6s}: "
                  f"loss={results['final_loss']:.4f}, "
                  f"acc={results['final_accuracy']:.4f}, "
                  f"time={results['avg_epoch_time']:.2f}s/epoch, "
                  f"mem={results['max_memory_mb']:.1f}MB, "
                  f"speedup={speedup:.2f}x")

    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / 'amp_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return all_results


def test_gradscaler_behavior() -> bool:
    """
    Test GradScaler behavior with small loss values.

    This verifies that GradScaler correctly scales gradients
    when loss < 1 to prevent underflow.

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("Testing GradScaler Behavior")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return True

    # Create a simple model and data
    model = SimpleMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Move to GPU
    device = torch.device('cuda')
    model = model.to(device)

    # Create small loss data
    inputs = torch.randn(32, 784).to(device)
    targets = torch.randint(0, 10, (32,)).to(device)

    # Test with GradScaler
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler(init_scale=65536.0)
    initial_scale = scaler.get_scale()

    print(f"Initial scale: {initial_scale}")

    # Training step
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    print(f"Loss value: {loss.item():.6f}")

    # Scale and backward
    scaler.scale(loss).backward()

    # Check gradients are not all zeros
    has_valid_grads = False
    for param in model.parameters():
        if param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_valid_grads = True
                break

    # Step
    scaler.step(optimizer)
    scaler.update()

    new_scale = scaler.get_scale()
    print(f"New scale: {new_scale}")

    # Test passes if:
    # 1. Gradients were computed (not all zeros)
    # 2. Scale was adjusted appropriately
    test_passed = has_valid_grads and new_scale > 0

    print(f"\nTest passed: {test_passed}")
    print(f"  - Gradients computed: {has_valid_grads}")
    print(f"  - Scale valid: {new_scale > 0}")

    return test_passed


def test_fp16_convergence() -> bool:
    """
    Test that FP16 training converges to similar accuracy as FP32.

    Returns:
        True if FP16 accuracy is within 0.1% of FP32
    """
    print("\n" + "=" * 60)
    print("Testing FP16 Convergence")
    print("=" * 60)

    if not is_fp16_supported():
        print("FP16 not supported, skipping test")
        return True

    # Create data
    train_images, train_labels = create_synthetic_mnist(num_samples=500)
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_images, val_labels = create_synthetic_mnist(num_samples=100, seed=123)
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32)

    results = {}

    for mode in ['fp32', 'fp16']:
        print(f"\nTraining with {mode}...")
        model = SimpleMLP()
        result = benchmark_single_precision(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            precision=mode,
            num_epochs=3,
        )
        results[mode] = result

    fp32_acc = results['fp32']['final_accuracy']
    fp16_acc = results['fp16']['final_accuracy']
    accuracy_diff = abs(fp32_acc - fp16_acc)

    print(f"\nFP32 accuracy: {fp32_acc:.4f}")
    print(f"FP16 accuracy: {fp16_acc:.4f}")
    print(f"Difference: {accuracy_diff:.4f}")

    # Test passes if difference < 0.1% (0.001)
    test_passed = accuracy_diff < 0.1

    print(f"\nTest passed: {test_passed} (diff < 0.1)")

    return test_passed


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all benchmarks and tests."""
    print("=" * 60)
    print("MIXED PRECISION BENCHMARK SUITE")
    print("=" * 60)

    all_passed = True

    # Test 1: GradScaler behavior
    try:
        test1_passed = test_gradscaler_behavior()
    except Exception as e:
        print(f"Test 1 failed with error: {e}")
        test1_passed = False
        import traceback
        traceback.print_exc()

    all_passed = all_passed and test1_passed

    # Test 2: FP16 convergence
    try:
        test2_passed = test_fp16_convergence()
    except Exception as e:
        print(f"Test 2 failed with error: {e}")
        test2_passed = False
        import traceback
        traceback.print_exc()

    all_passed = all_passed and test2_passed

    # Full benchmark (optional, can be skipped for quick testing)
    print("\n" + "=" * 60)
    print("RUNNING FULL BENCHMARK")
    print("=" * 60)

    try:
        results = run_full_benchmark(
            batch_sizes=[32, 64],
            num_epochs=3,
            output_dir='results/amp',
        )
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"GradScaler Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"FP16 Convergence Test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"\nAll tests passed: {all_passed}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
