"""
Image Augmentation: Data Augmentation Techniques for Computer Vision.

This module provides image augmentation implementations:
    - Geometric: RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
    - Color: ColorJitter (brightness, contrast, saturation, hue)
    - Advanced: Mixup, CutMix, RandomErasing

Theory:
    Data Augmentation:
        - Increases effective training data diversity
        - Acts as regularization, reducing overfitting
        - Improves model generalization

    Mixup:
        - Creates virtual training examples by linear interpolation
        - x̃ = λx_i + (1-λ)x_j, ỹ = λy_i + (1-λ)y_j
        - λ ~ Beta(α, α), encourages linear behavior between classes

    CutMix:
        - Cuts a patch from one image and pastes onto another
        - Ground truth labels mixed proportionally to patch area
        - Better localization than Mixup

    RandomErasing:
        - Randomly selects a rectangle region and erases pixels
        - Simulates occlusion, improves robustness

References:
    - mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    - CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    - Random Erasing Data Augmentation (Zhong et al., 2017)
"""

from typing import Tuple, Optional, Union, List, Callable, Dict, Any
from dataclasses import dataclass, field
import numpy as np

ArrayLike = Union[np.ndarray, List, float]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array with float64 dtype."""
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    elif x.dtype != np.float64:
        x = x.astype(np.float64)
    return x


# =============================================================================
# Geometric Augmentations
# =============================================================================


class RandomCrop:
    """
    Random cropping augmentation.

    Randomly crops the image to a specified size. If the crop size is larger
    than the image, padding is applied first.

    Args:
        crop_size: Target crop size (height, width) or single int for square
        padding: Padding to apply before cropping (optional)
        pad_mode: Padding mode ('constant', 'reflect', 'edge', 'wrap')
        pad_value: Value for constant padding

    Example:
        >>> crop = RandomCrop(crop_size=(24, 24), padding=4)
        >>> image = np.random.rand(32, 32, 3)
        >>> cropped = crop(image)
        >>> cropped.shape
        (24, 24, 3)
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        padding: Optional[int] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ):
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = crop_size

        self.padding = padding
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        # Random state for reproducibility
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random crop to image.

        Args:
            image: Input image (H, W, C) or (H, W)
            seed: Random seed for reproducibility

        Returns:
            Cropped image
        """
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Handle grayscale
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        h, w = image.shape[:2]

        # Apply padding if specified
        if self.padding is not None:
            pad = self.padding
            if self.pad_mode == "constant":
                image = np.pad(
                    image,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode="constant",
                    constant_values=self.pad_value,
                )
            else:
                image = np.pad(
                    image, ((pad, pad), (pad, pad), (0, 0)), mode=self.pad_mode
                )
            h, w = image.shape[:2]

        # Check if crop is valid
        if self.crop_h > h or self.crop_w > w:
            raise ValueError(
                f"Crop size ({self.crop_h}, {self.crop_w}) larger than "
                f"image size ({h}, {w})"
            )

        # Random crop position
        top = self._rng.integers(0, h - self.crop_h + 1)
        left = self._rng.integers(0, w - self.crop_w + 1)

        # Crop
        cropped = image[top : top + self.crop_h, left : left + self.crop_w]

        if squeeze_output:
            cropped = cropped[:, :, 0]

        return cropped

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomHorizontalFlip:
    """
    Random horizontal flip augmentation.

    Flips the image horizontally with probability p.

    Args:
        p: Probability of flipping (default: 0.5)

    Example:
        >>> flip = RandomHorizontalFlip(p=0.5)
        >>> image = np.arange(6).reshape(2, 3, 1)
        >>> flipped = flip(image)
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random horizontal flip.

        Args:
            image: Input image (H, W, C) or (H, W)
            seed: Random seed

        Returns:
            Potentially flipped image
        """
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._rng.random() < self.p:
            return np.fliplr(image).copy()
        return image.copy()

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomVerticalFlip:
    """
    Random vertical flip augmentation.

    Flips the image vertically with probability p.

    Args:
        p: Probability of flipping (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        self.p = p
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply random vertical flip."""
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._rng.random() < self.p:
            return np.flipud(image).copy()
        return image.copy()

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomRotation:
    """
    Random rotation augmentation.

    Rotates the image by a random angle within the specified range.
    Uses nearest neighbor interpolation for simplicity.

    Args:
        degrees: Range of rotation in degrees (min, max) or single value for [-deg, deg]
        fill: Value to fill empty areas after rotation

    Example:
        >>> rotate = RandomRotation(degrees=15)
        >>> image = np.random.rand(32, 32, 3)
        >>> rotated = rotate(image)
    """

    def __init__(
        self, degrees: Union[float, Tuple[float, float]], fill: float = 0.0
    ):
        if isinstance(degrees, (int, float)):
            self.min_deg, self.max_deg = -degrees, degrees
        else:
            self.min_deg, self.max_deg = degrees

        self.fill = fill
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random rotation.

        Args:
            image: Input image (H, W, C) or (H, W)
            seed: Random seed

        Returns:
            Rotated image
        """
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        angle = self._rng.uniform(self.min_deg, self.max_deg)
        angle_rad = np.deg2rad(angle)

        h, w = image.shape[:2]
        has_channel = image.ndim == 3
        c = image.shape[2] if has_channel else 1

        # Center of rotation
        cx, cy = w / 2, h / 2

        # Create output image
        output = np.full_like(image, self.fill)

        # Rotation matrix
        cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)

        # For each pixel in output, find source pixel
        for y_out in range(h):
            for x_out in range(w):
                # Translate to center
                x_c = x_out - cx
                y_c = y_out - cy

                # Rotate (inverse transform)
                x_src = cos_a * x_c - sin_a * y_c + cx
                y_src = sin_a * x_c + cos_a * y_c + cy

                # Nearest neighbor
                x_src = int(round(x_src))
                y_src = int(round(y_src))

                # Check bounds
                if 0 <= x_src < w and 0 <= y_src < h:
                    if has_channel:
                        output[y_out, x_out] = image[y_src, x_src]
                    else:
                        output[y_out, x_src] = image[y_src, x_src]

        return output

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Color Augmentations
# =============================================================================


class ColorJitter:
    """
    Random color jitter augmentation.

    Randomly adjusts brightness, contrast, saturation, and hue.

    Args:
        brightness: Brightness adjustment range (factor or (min, max))
        contrast: Contrast adjustment range (factor or (min, max))
        saturation: Saturation adjustment range (factor or (min, max))
        hue: Hue adjustment range (-0.5 to 0.5)

    Example:
        >>> jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        >>> image = np.random.rand(32, 32, 3)
        >>> jittered = jitter(image)
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.0,
        contrast: Union[float, Tuple[float, float]] = 0.0,
        saturation: Union[float, Tuple[float, float]] = 0.0,
        hue: Union[float, Tuple[float, float]] = 0.0,
    ):
        self.brightness = self._parse_range(brightness, "brightness")
        self.contrast = self._parse_range(contrast, "contrast")
        self.saturation = self._parse_range(saturation, "saturation")
        self.hue = self._parse_range(hue, "hue", max_val=0.5)
        self._rng = np.random.default_rng()

    def _parse_range(
        self, value: Union[float, Tuple[float, float]], name: str, max_val: float = 1.0
    ) -> Tuple[float, float]:
        """Parse range parameter."""
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} factor must be non-negative")
            return (1 - value, 1 + value)
        else:
            min_v, max_v = value
            if min_v < 0:
                raise ValueError(f"{name} min must be non-negative")
            return (min_v, max_v)

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply color jitter.

        Args:
            image: Input RGB image (H, W, 3), assumed to be in [0, 1] or [0, 255]
            seed: Random seed

        Returns:
            Color-jittered image
        """
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("ColorJitter expects RGB images with shape (H, W, 3)")

        # Detect range
        is_normalized = image.max() <= 1.0

        # Apply augmentations in order
        output = image.copy()

        # Brightness
        if self.brightness != (1.0, 1.0):
            factor = self._rng.uniform(*self.brightness)
            output = output * factor

        # Contrast
        if self.contrast != (1.0, 1.0):
            factor = self._rng.uniform(*self.contrast)
            mean = output.mean()
            output = (output - mean) * factor + mean

        # Saturation (via grayscale)
        if self.saturation != (1.0, 1.0):
            factor = self._rng.uniform(*self.saturation)
            gray = self._rgb_to_grayscale(output)
            gray = np.stack([gray, gray, gray], axis=-1)
            output = output * factor + gray * (1 - factor)

        # Hue (via HSV conversion)
        if self.hue != (0.0, 0.0):
            factor = self._rng.uniform(*self.hue)
            output = self._adjust_hue(output, factor)

        # Clip to valid range
        if is_normalized:
            output = np.clip(output, 0, 1)
        else:
            output = np.clip(output, 0, 255)

        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale."""
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

    def _adjust_hue(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust hue of RGB image."""
        # Simple hue adjustment via channel rotation
        # This is a simplified version - full implementation would use HSV
        h_shift = int(factor * 3) % 3
        if h_shift == 0:
            return image
        elif h_shift == 1:
            return image[:, :, [2, 0, 1]]
        else:
            return image[:, :, [1, 2, 0]]

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Advanced Augmentations
# =============================================================================


class Mixup:
    """
    Mixup data augmentation.

    Creates virtual training examples by linear interpolation between
    pairs of examples and their labels.

    Formula:
        x̃ = λ * x_i + (1 - λ) * x_j
        ỹ = λ * y_i + (1 - λ) * y_j
        λ ~ Beta(α, α)

    Args:
        alpha: Beta distribution parameter (default: 1.0)
        inplace: Whether to modify inputs in place

    Example:
        >>> mixup = Mixup(alpha=1.0)
        >>> images = np.random.rand(32, 3, 32, 32)
        >>> labels = np.eye(10)[np.random.randint(0, 10, 32)]  # one-hot
        >>> mixed_images, mixed_labels = mixup(images, labels)

    References:
        - mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    """

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self._rng = np.random.default_rng()

    def __call__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to a batch of images and labels.

        Args:
            images: Batch of images (N, C, H, W) or (N, H, W, C)
            labels: Batch of labels (N,) for class indices or (N, C) for one-hot
            seed: Random seed

        Returns:
            Tuple of (mixed_images, mixed_labels)
        """
        images = _ensure_array(images)
        labels = np.asarray(labels)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        batch_size = images.shape[0]

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = self._rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Shuffle indices
        idx = self._rng.permutation(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[idx]

        # Mix labels
        if labels.ndim == 1:
            # Class indices - convert to one-hot
            num_classes = labels.max() + 1
            labels_onehot = np.eye(num_classes)[labels]
            labels_shuffled = np.eye(num_classes)[labels[idx]]
            mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled
        else:
            # Already one-hot or soft labels
            mixed_labels = lam * labels + (1 - lam) * labels[idx]

        return mixed_images, mixed_labels

    def get_lambda(self, seed: Optional[int] = None) -> float:
        """Sample a mixing coefficient."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self._rng.beta(self.alpha, self.alpha)

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class CutMix:
    """
    CutMix data augmentation.

    Cuts a patch from one image and pastes it onto another.
    Labels are mixed proportionally to the patch area.

    Formula:
        Bbox ~ Uniform(0, W), Uniform(0, H)
        Area ratio = λ
        x̃ = M ⊙ x_i + (1-M) ⊙ x_j
        ỹ = λ * y_i + (1 - λ) * y_j

    Args:
        alpha: Beta distribution parameter for λ sampling (default: 1.0)
        min_area: Minimum cut area ratio (default: 0.02)

    Example:
        >>> cutmix = CutMix(alpha=1.0)
        >>> images = np.random.rand(32, 3, 32, 32)
        >>> labels = np.eye(10)[np.random.randint(0, 10, 32)]
        >>> mixed_images, mixed_labels = cutmix(images, labels)

    References:
        - CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    """

    def __init__(self, alpha: float = 1.0, min_area: float = 0.02):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.min_area = min_area
        self._rng = np.random.default_rng()

    def __call__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix to a batch of images and labels.

        Args:
            images: Batch of images (N, C, H, W) or (N, H, W, C)
            labels: Batch of labels (N,) or (N, num_classes)
            seed: Random seed

        Returns:
            Tuple of (mixed_images, mixed_labels)
        """
        images = _ensure_array(images)
        labels = np.asarray(labels)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        batch_size = images.shape[0]

        # Determine image format
        if images.shape[1] <= 4:  # (N, C, H, W)
            h, w = images.shape[2], images.shape[3]
            channel_first = True
        else:  # (N, H, W, C)
            h, w = images.shape[1], images.shape[2]
            channel_first = False

        # Sample lambda from Beta distribution
        lam = self._rng.beta(self.alpha, self.alpha)

        # Compute cut box size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Random center point
        cx = self._rng.integers(0, w)
        cy = self._rng.integers(0, h)

        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Adjust lambda to actual area ratio
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = w * h
        lam = 1.0 - actual_area / total_area

        # Shuffle indices
        idx = self._rng.permutation(batch_size)

        # Mix images
        mixed_images = images.copy()
        if channel_first:
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
                idx, :, bby1:bby2, bbx1:bbx2
            ]
        else:
            mixed_images[:, bby1:bby2, bbx1:bbx2, :] = images[
                idx, bby1:bby2, bbx1:bbx2, :
            ]

        # Mix labels
        if labels.ndim == 1:
            num_classes = labels.max() + 1
            labels_onehot = np.eye(num_classes)[labels]
            labels_shuffled = np.eye(num_classes)[labels[idx]]
            mixed_labels = lam * labels_onehot + (1 - lam) * labels_shuffled
        else:
            mixed_labels = lam * labels + (1 - lam) * labels[idx]

        return mixed_images, mixed_labels

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomErasing:
    """
    Random erasing augmentation.

    Randomly selects a rectangle region in an image and erases its pixels.
    This simulates occlusion and improves model robustness.

    Args:
        probability: Probability of applying erasing (default: 0.5)
        scale_range: Range of erasing area ratio (min, max) (default: (0.02, 0.33))
        ratio_range: Range of aspect ratio (min, max) (default: (0.3, 3.3))
        value: Erasing value - 'random', 'mean', or a number (default: 'random')
        inplace: Whether to modify image in place

    Example:
        >>> eraser = RandomErasing(probability=0.5, value='random')
        >>> image = np.random.rand(3, 32, 32)
        >>> erased = eraser(image)

    References:
        - Random Erasing Data Augmentation (Zhong et al., 2017)
    """

    def __init__(
        self,
        probability: float = 0.5,
        scale_range: Tuple[float, float] = (0.02, 0.33),
        ratio_range: Tuple[float, float] = (0.3, 3.3),
        value: Union[str, float] = "random",
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        self.probability = probability
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.value = value
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random erasing.

        Args:
            image: Input image (C, H, W) or (H, W, C) or (H, W)
            seed: Random seed

        Returns:
            Image with potentially erased region
        """
        image = _ensure_array(image)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._rng.random() > self.probability:
            return image.copy()

        # Determine dimensions
        need_transpose = False
        squeeze_output = False

        if image.ndim == 2:
            h, w = image.shape
            c = 1
            image = image[:, :, np.newaxis]
            squeeze_output = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            image = np.transpose(image, (1, 2, 0))
            need_transpose = True
        else:  # (H, W, C)
            h, w, c = image.shape

        # Calculate erasing area
        area = h * w
        target_area = self._rng.uniform(*self.scale_range) * area
        aspect_ratio = self._rng.uniform(*self.ratio_range)

        # Calculate h and w of erasing region
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        # Retry if too large
        if h_erase >= h or w_erase >= w:
            if need_transpose:
                return np.transpose(image, (2, 0, 1))
            elif squeeze_output:
                return image[:, :, 0]
            else:
                return image.copy()

        # Random position
        y = self._rng.integers(0, h - h_erase)
        x = self._rng.integers(0, w - w_erase)

        # Erase region
        output = image.copy()
        if self.value == "random":
            output[y : y + h_erase, x : x + w_erase, :] = self._rng.random(
                (h_erase, w_erase, c)
            )
        elif self.value == "mean":
            mean_val = image.mean(axis=(0, 1), keepdims=True)
            output[y : y + h_erase, x : x + w_erase, :] = mean_val
        else:
            output[y : y + h_erase, x : x + w_erase, :] = self.value

        # Restore format
        if need_transpose:
            output = np.transpose(output, (2, 0, 1))
        if squeeze_output:
            output = output[:, :, 0]

        return output

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Composite Transform
# =============================================================================


class Compose:
    """
    Compose multiple augmentations into a single transform.

    Args:
        transforms: List of augmentation transforms

    Example:
        >>> transform = Compose([
        ...     RandomCrop(24, padding=4),
        ...     RandomHorizontalFlip(p=0.5),
        ...     ColorJitter(brightness=0.4, contrast=0.4),
        ... ])
        >>> image = np.random.rand(32, 32, 3)
        >>> augmented = transform(image)
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Apply all transforms sequentially."""
        output = image
        rng = np.random.default_rng(seed) if seed is not None else None

        for i, t in enumerate(self.transforms):
            if rng is not None and hasattr(t, "set_rng"):
                t_seed = rng.integers(0, 2**31)
                output = t(output, seed=t_seed)
            else:
                output = t(output)

        return output

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


# =============================================================================
# Registry
# =============================================================================


IMAGE_AUGMENTATIONS: Dict[str, type] = {
    "random_crop": RandomCrop,
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_vertical_flip": RandomVerticalFlip,
    "random_rotation": RandomRotation,
    "color_jitter": ColorJitter,
    "mixup": Mixup,
    "cutmix": CutMix,
    "random_erasing": RandomErasing,
    "compose": Compose,
}


def get_augmentation(name: str, **kwargs) -> Callable:
    """
    Get augmentation by name.

    Args:
        name: Augmentation name (case-insensitive)
        **kwargs: Arguments to pass to augmentation constructor

    Returns:
        Augmentation instance

    Raises:
        ValueError: If augmentation name is not found

    Example:
        >>> aug = get_augmentation("random_crop", crop_size=24)
        >>> isinstance(aug, RandomCrop)
        True
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower not in IMAGE_AUGMENTATIONS:
        available = list(IMAGE_AUGMENTATIONS.keys())
        raise ValueError(
            f"Unknown augmentation '{name}'. Available: {available}"
        )

    return IMAGE_AUGMENTATIONS[name_lower](**kwargs)


def list_augmentations() -> List[str]:
    """List all available augmentations."""
    return list(IMAGE_AUGMENTATIONS.keys())
