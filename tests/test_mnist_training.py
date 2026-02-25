"""
MNIST-style training test for loss functions.

Tests that loss functions can train on synthetic MNIST-like data for 1 epoch and converge.
Uses synthetic data to avoid downloading MNIST dataset.

Run: pytest tests/test_mnist_training.py -v
"""

import pytest
import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_synthetic_mnist(n_samples=1000, img_size=28, n_classes=10, seed=42):
    """
    Create synthetic MNIST-like data for testing.

    This creates random images with class labels, useful for testing
    loss functions without downloading real MNIST data.

    Args:
        n_samples: Number of samples to generate
        img_size: Size of square images
        n_classes: Number of classes
        seed: Random seed for reproducibility

    Returns:
        DataLoader with synthetic data
    """
    torch.manual_seed(seed)

    # Generate random images (similar to MNIST statistics)
    # MNIST has mean ~0.1307 and std ~0.3081 after normalization
    data = torch.randn(n_samples, 1, img_size, img_size) * 0.3081 + 0.1307

    # Generate random labels
    targets = torch.randint(0, n_classes, (n_samples,))

    # Create dataset and dataloader
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestMNISTTraining:
    """Test loss functions on MNIST-style training."""

    @pytest.fixture(scope="class")
    def mnist_data(self):
        """Create synthetic MNIST-like dataset."""
        return create_synthetic_mnist(n_samples=1000)

    @pytest.fixture
    def simple_mlp(self):
        """Create simple MLP for MNIST."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        return model

    def test_mse_loss_converges(self, mnist_data, simple_mlp):
        """Test MSE loss converges on MNIST."""
        model = simple_mlp
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for batch_idx, (data, target) in enumerate(mnist_data):
            optimizer.zero_grad()
            output = model(data)

            # Convert target to one-hot for MSE
            target_one_hot = torch.zeros(data.size(0), 10)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)

            loss = criterion(output, target_one_hot)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that loss decreased
        avg_first_half = np.mean(losses[:len(losses)//2])
        avg_second_half = np.mean(losses[len(losses)//2:])
        assert avg_second_half < avg_first_half, "MSE loss should decrease during training"

    def test_cross_entropy_loss_converges(self, mnist_data, simple_mlp):
        """Test cross-entropy loss converges on MNIST."""
        model = simple_mlp
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for batch_idx, (data, target) in enumerate(mnist_data):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that loss decreased
        avg_first_half = np.mean(losses[:len(losses)//2])
        avg_second_half = np.mean(losses[len(losses)//2:])
        assert avg_second_half < avg_first_half, "Cross-entropy loss should decrease"

    def test_cross_entropy_with_label_smoothing(self, mnist_data, simple_mlp):
        """Test cross-entropy with label smoothing works without error."""
        model = simple_mlp
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        losses = []
        for batch_idx, (data, target) in enumerate(mnist_data):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify loss is finite and reasonable
        assert all(np.isfinite(losses)), "Loss should be finite"
        assert np.mean(losses) < 5.0, "Loss should be reasonable (<5)"

    def test_focal_loss_converges(self, mnist_data, simple_mlp):
        """Test focal loss converges on MNIST-style data."""
        model = simple_mlp
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Implement focal loss manually
        def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
            ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** gamma
            loss = alpha * focal_weight * ce_loss
            return loss.mean()

        losses = []
        for batch_idx, (data, target) in enumerate(mnist_data):
            optimizer.zero_grad()
            output = model(data)
            loss = focal_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that loss decreased
        avg_first_half = np.mean(losses[:len(losses)//2])
        avg_second_half = np.mean(losses[len(losses)//2:])
        assert avg_second_half < avg_first_half, "Focal loss should decrease"

    def test_loss_functions_run_successfully(self, mnist_data):
        """Test that all loss functions can run a training loop without errors."""
        losses_tested = []

        # Test MSE
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for data, target in mnist_data:
            optimizer.zero_grad()
            output = model(data)
            target_one_hot = torch.zeros(data.size(0), 10)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            loss = nn.MSELoss()(output, target_one_hot)
            loss.backward()
            optimizer.step()
            break
        losses_tested.append("MSE")

        # Test CrossEntropy
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for data, target in mnist_data:
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            break
        losses_tested.append("CrossEntropy")

        # Test CrossEntropy with label smoothing
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for data, target in mnist_data:
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(output, target)
            loss.backward()
            optimizer.step()
            break
        losses_tested.append("CrossEntropy+LabelSmoothing")

        # Test Triplet Loss
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        for data, target in mnist_data:
            # Create triplet from batch
            if data.size(0) >= 3:
                anchor = data[0:1]
                positive = data[1:2]  # Different sample, same class not guaranteed
                negative = data[2:3]
                optimizer.zero_grad()
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = nn.TripletMarginLoss(margin=0.5)(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()
            break
        losses_tested.append("Triplet")

        assert len(losses_tested) == 4, f"Expected 4 loss functions, tested: {losses_tested}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
