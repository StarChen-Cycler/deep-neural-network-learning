"""
Text Augmentation: Data Augmentation Techniques for NLP.

This module provides text augmentation implementations:
    - Token masking (BERT-style)
    - Random token replacement
    - Token deletion
    - Synonym replacement

Theory:
    Token Masking (BERT):
        - Randomly mask tokens with [MASK] token
        - 15% of tokens selected, 80% masked, 10% random, 10% unchanged
        - Forces model to learn contextual representations

    Data Augmentation for NLP:
        - Limited compared to vision due to discrete nature of text
        - Small changes can drastically alter meaning
        - Augmentation must preserve semantic meaning

References:
    - BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
    - EDA: Easy Data Augmentation Techniques for Boosting Text Classification (Wei & Zou, 2019)
"""

from typing import Tuple, Optional, Union, List, Callable, Dict, Any
from dataclasses import dataclass, field
import numpy as np

ArrayLike = Union[np.ndarray, List, float]


def _ensure_array(x: ArrayLike) -> np.ndarray:
    """Ensure input is a numpy array."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


# =============================================================================
# Token Masking (BERT-style)
# =============================================================================


class TokenMasking:
    """
    BERT-style token masking for self-supervised pretraining.

    Randomly masks tokens for masked language modeling (MLM).
    Following BERT paper:
        - 15% of tokens selected
        - 80% replaced with [MASK]
        - 10% replaced with random token
        - 10% unchanged

    Args:
        mask_prob: Probability of masking a token (default: 0.15)
        mask_token_id: Token ID for [MASK] (default: 0, set to your vocab)
        random_token_prob: Probability of replacing with random token (default: 0.1)
        unchanged_prob: Probability of leaving token unchanged (default: 0.1)
        vocab_size: Size of vocabulary for random token replacement
        special_token_ids: IDs of special tokens to never mask (CLS, SEP, PAD, etc.)

    Example:
        >>> masker = TokenMasking(mask_prob=0.15, mask_token_id=103, vocab_size=30522)
        >>> tokens = np.array([101, 2023, 2003, 1037, 3231, 102])  # [CLS] this is a test [SEP]
        >>> masked, labels = masker(tokens)
        >>> # masked may have some tokens replaced with 103 (MASK)
        >>> # labels contains original tokens for loss computation
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 0,
        random_token_prob: float = 0.1,
        unchanged_prob: float = 0.1,
        vocab_size: Optional[int] = None,
        special_token_ids: Optional[List[int]] = None,
    ):
        if not 0 <= mask_prob <= 1:
            raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
        if random_token_prob + unchanged_prob > 1.0:
            raise ValueError(
                f"random_token_prob + unchanged_prob must be <= 1, "
                f"got {random_token_prob + unchanged_prob}"
            )

        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.random_token_prob = random_token_prob
        self.unchanged_prob = unchanged_prob
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids) if special_token_ids else set()
        self._rng = np.random.default_rng()

    def __call__(
        self,
        tokens: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply token masking.

        Args:
            tokens: Input token IDs (seq_len,) or (batch, seq_len)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (masked_tokens, labels)
            - masked_tokens: Tokens with masking applied
            - labels: Original tokens (-100 for non-masked positions)
        """
        tokens = _ensure_array(tokens)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        original_shape = tokens.shape
        is_batched = tokens.ndim == 2

        if not is_batched:
            tokens = tokens[np.newaxis, :]

        batch_size, seq_len = tokens.shape
        masked_tokens = tokens.copy()
        labels = np.full_like(tokens, -100)  # -100 is ignored in cross-entropy

        for b in range(batch_size):
            # Find maskable positions (exclude special tokens)
            maskable = np.array(
                [i for i, t in enumerate(tokens[b]) if t not in self.special_token_ids]
            )

            if len(maskable) == 0:
                continue

            # Handle case where mask_prob is 0
            if self.mask_prob == 0:
                continue

            # Number of tokens to select
            n_mask = max(1, int(len(maskable) * self.mask_prob))

            # Random selection
            selected = self._rng.choice(maskable, size=min(n_mask, len(maskable)), replace=False)

            for idx in selected:
                # Save original token as label
                labels[b, idx] = tokens[b, idx]

                # Determine replacement strategy (BERT-style: 80% mask, 10% random, 10% unchanged)
                rand = self._rng.random()

                mask_threshold = 0.8  # 80% replace with [MASK]
                random_threshold = 0.9  # 10% replace with random token

                if rand < mask_threshold:
                    # Replace with [MASK]
                    masked_tokens[b, idx] = self.mask_token_id
                elif rand < random_threshold:
                    # Replace with random token
                    if self.vocab_size is not None:
                        masked_tokens[b, idx] = self._rng.integers(0, self.vocab_size)
                    else:
                        masked_tokens[b, idx] = self.mask_token_id
                # else: leave unchanged (10%)

        if not is_batched:
            masked_tokens = masked_tokens[0]
            labels = labels[0]

        return masked_tokens, labels

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomTokenMasking:
    """
    Simpler random token masking for text augmentation.

    Randomly masks a percentage of tokens without BERT's 80/10/10 rule.
    Useful for text classification augmentation.

    Args:
        mask_prob: Probability of masking each token
        mask_token_id: Token ID for [MASK] or replacement token
        special_token_ids: IDs of special tokens to never mask

    Example:
        >>> masker = RandomTokenMasking(mask_prob=0.1, mask_token_id=0)
        >>> tokens = np.array([1, 2, 3, 4, 5])
        >>> masked = masker(tokens)
    """

    def __init__(
        self,
        mask_prob: float = 0.1,
        mask_token_id: int = 0,
        special_token_ids: Optional[List[int]] = None,
    ):
        if not 0 <= mask_prob <= 1:
            raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")

        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.special_token_ids = set(special_token_ids) if special_token_ids else set()
        self._rng = np.random.default_rng()

    def __call__(
        self,
        tokens: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply random masking.

        Args:
            tokens: Input token IDs (seq_len,) or (batch, seq_len)
            seed: Random seed

        Returns:
            Masked tokens
        """
        tokens = _ensure_array(tokens)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        original_shape = tokens.shape
        is_batched = tokens.ndim == 2

        if not is_batched:
            tokens = tokens[np.newaxis, :]

        masked_tokens = tokens.copy()
        batch_size, seq_len = tokens.shape

        for b in range(batch_size):
            for i in range(seq_len):
                if tokens[b, i] not in self.special_token_ids:
                    if self._rng.random() < self.mask_prob:
                        masked_tokens[b, i] = self.mask_token_id

        if not is_batched:
            masked_tokens = masked_tokens[0]

        return masked_tokens

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Token Replacement
# =============================================================================


class RandomTokenReplacement:
    """
    Random token replacement augmentation.

    Randomly replaces tokens with other tokens from vocabulary.
    Similar to BERT's random replacement but can be used independently.

    Args:
        replace_prob: Probability of replacing each token
        vocab_size: Size of vocabulary for replacement sampling
        special_token_ids: IDs of special tokens to never replace

    Example:
        >>> replacer = RandomTokenReplacement(replace_prob=0.1, vocab_size=10000)
        >>> tokens = np.array([1, 2, 3, 4, 5])
        >>> replaced = replacer(tokens)
    """

    def __init__(
        self,
        replace_prob: float = 0.1,
        vocab_size: int = 30522,
        special_token_ids: Optional[List[int]] = None,
    ):
        if not 0 <= replace_prob <= 1:
            raise ValueError(f"replace_prob must be in [0, 1], got {replace_prob}")

        self.replace_prob = replace_prob
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids) if special_token_ids else set()
        self._rng = np.random.default_rng()

    def __call__(
        self,
        tokens: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply random token replacement.

        Args:
            tokens: Input token IDs (seq_len,) or (batch, seq_len)
            seed: Random seed

        Returns:
            Tokens with some replaced
        """
        tokens = _ensure_array(tokens)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        original_shape = tokens.shape
        is_batched = tokens.ndim == 2

        if not is_batched:
            tokens = tokens[np.newaxis, :]

        replaced_tokens = tokens.copy()
        batch_size, seq_len = tokens.shape

        for b in range(batch_size):
            for i in range(seq_len):
                if tokens[b, i] not in self.special_token_ids:
                    if self._rng.random() < self.replace_prob:
                        # Sample random token (avoid special tokens if possible)
                        new_token = self._rng.integers(0, self.vocab_size)
                        while new_token in self.special_token_ids and self.vocab_size > len(self.special_token_ids):
                            new_token = self._rng.integers(0, self.vocab_size)
                        replaced_tokens[b, i] = new_token

        if not is_batched:
            replaced_tokens = replaced_tokens[0]

        return replaced_tokens

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Token Operations
# =============================================================================


class RandomTokenDeletion:
    """
    Random token deletion augmentation.

    Randomly deletes tokens from the sequence.
    Useful for making models robust to missing words.

    Args:
        delete_prob: Probability of deleting each token
        special_token_ids: IDs of special tokens to never delete
        max_delete_ratio: Maximum ratio of tokens to delete (default: 0.3)

    Example:
        >>> deleter = RandomTokenDeletion(delete_prob=0.1)
        >>> tokens = np.array([1, 2, 3, 4, 5])
        >>> deleted = deleter(tokens)
    """

    def __init__(
        self,
        delete_prob: float = 0.1,
        special_token_ids: Optional[List[int]] = None,
        max_delete_ratio: float = 0.3,
    ):
        if not 0 <= delete_prob <= 1:
            raise ValueError(f"delete_prob must be in [0, 1], got {delete_prob}")

        self.delete_prob = delete_prob
        self.special_token_ids = set(special_token_ids) if special_token_ids else set()
        self.max_delete_ratio = max_delete_ratio
        self._rng = np.random.default_rng()

    def __call__(
        self,
        tokens: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply random token deletion.

        Args:
            tokens: Input token IDs (seq_len,) or (batch, seq_len)
            seed: Random seed

        Returns:
            Tokens with some deleted (padded to original length)
        """
        tokens = _ensure_array(tokens)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        is_batched = tokens.ndim == 2

        if not is_batched:
            tokens = tokens[np.newaxis, :]

        batch_size, seq_len = tokens.shape
        results = []

        for b in range(batch_size):
            kept_tokens = []
            n_deleted = 0
            max_delete = int(seq_len * self.max_delete_ratio)

            for i in range(seq_len):
                token = tokens[b, i]

                # Never delete special tokens
                if token in self.special_token_ids:
                    kept_tokens.append(token)
                    continue

                # Check if we've deleted enough
                if n_deleted >= max_delete:
                    kept_tokens.append(token)
                    continue

                # Random deletion
                if self._rng.random() >= self.delete_prob:
                    kept_tokens.append(token)
                else:
                    n_deleted += 1

            # Pad to original length with 0s
            result = np.zeros(seq_len, dtype=tokens.dtype)
            result[: len(kept_tokens)] = kept_tokens
            results.append(result)

        output = np.stack(results)

        if not is_batched:
            output = output[0]

        return output

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomTokenInsertion:
    """
    Random token insertion augmentation.

    Randomly inserts tokens into the sequence at random positions.

    Args:
        insert_prob: Probability of inserting at each position
        vocab_size: Size of vocabulary for insertion sampling
        max_insertions: Maximum number of insertions (default: 3)

    Example:
        >>> inserter = RandomTokenInsertion(insert_prob=0.1, vocab_size=10000)
        >>> tokens = np.array([1, 2, 3, 4, 5])
        >>> inserted = inserter(tokens)
    """

    def __init__(
        self,
        insert_prob: float = 0.1,
        vocab_size: int = 30522,
        max_insertions: int = 3,
    ):
        if not 0 <= insert_prob <= 1:
            raise ValueError(f"insert_prob must be in [0, 1], got {insert_prob}")

        self.insert_prob = insert_prob
        self.vocab_size = vocab_size
        self.max_insertions = max_insertions
        self._rng = np.random.default_rng()

    def __call__(
        self,
        tokens: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply random token insertion.

        Args:
            tokens: Input token IDs (seq_len,) or (batch, seq_len)
            seed: Random seed

        Returns:
            Tokens with some inserted (truncated to original length if needed)
        """
        tokens = _ensure_array(tokens)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        is_batched = tokens.ndim == 2
        if not is_batched:
            tokens = tokens[np.newaxis, :]

        batch_size, seq_len = tokens.shape
        results = []

        for b in range(batch_size):
            new_tokens = list(tokens[b])
            n_inserted = 0

            i = 0
            while i < len(new_tokens) and n_inserted < self.max_insertions:
                if self._rng.random() < self.insert_prob:
                    # Insert random token
                    new_token = self._rng.integers(0, self.vocab_size)
                    new_tokens.insert(i, new_token)
                    n_inserted += 1
                    i += 1  # Skip the inserted token
                i += 1

            # Truncate to original length
            if len(new_tokens) > seq_len:
                new_tokens = new_tokens[:seq_len]
            elif len(new_tokens) < seq_len:
                new_tokens = new_tokens + [0] * (seq_len - len(new_tokens))

            results.append(new_tokens)

        output = np.array(results, dtype=tokens.dtype)

        if not is_batched:
            output = output[0]

        return output

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Word-level Operations (for tokenized text)
# =============================================================================


class SynonymReplacement:
    """
    Synonym replacement augmentation.

    Replaces words with their synonyms. This requires a synonym dictionary.
    For demonstration, uses a simple placeholder approach.

    Args:
        replace_prob: Probability of replacing each word
        n_replacements: Maximum number of replacements
        synonym_dict: Dictionary mapping words to list of synonyms

    Example:
        >>> synonyms = {"good": ["great", "excellent", "fine"], "bad": ["poor", "terrible"]}
        >>> replacer = SynonymReplacement(replace_prob=0.3, synonym_dict=synonyms)
        >>> words = ["this", "is", "good"]
        >>> replaced = replacer(words)
    """

    def __init__(
        self,
        replace_prob: float = 0.3,
        n_replacements: int = 1,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
    ):
        if not 0 <= replace_prob <= 1:
            raise ValueError(f"replace_prob must be in [0, 1], got {replace_prob}")

        self.replace_prob = replace_prob
        self.n_replacements = n_replacements
        self.synonym_dict = synonym_dict or {}
        self._rng = np.random.default_rng()

    def __call__(
        self,
        words: Union[List[str], np.ndarray],
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Apply synonym replacement.

        Args:
            words: List of words
            seed: Random seed

        Returns:
            Words with some replaced by synonyms
        """
        if isinstance(words, np.ndarray):
            words = words.tolist()

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        result = list(words)
        n_replaced = 0

        indices = list(range(len(result)))
        self._rng.shuffle(indices)

        for i in indices:
            if n_replaced >= self.n_replacements:
                break

            word = result[i].lower()

            if word in self.synonym_dict and self._rng.random() < self.replace_prob:
                synonyms = self.synonym_dict[word]
                if synonyms:
                    result[i] = self._rng.choice(synonyms)
                    n_replaced += 1

        return result

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


class RandomSwap:
    """
    Random word swap augmentation.

    Randomly swaps adjacent words in the sequence.

    Args:
        n_swaps: Number of swaps to perform (default: 1)

    Example:
        >>> swapper = RandomSwap(n_swaps=2)
        >>> words = ["the", "quick", "brown", "fox"]
        >>> swapped = swapper(words)
    """

    def __init__(self, n_swaps: int = 1):
        if n_swaps < 0:
            raise ValueError(f"n_swaps must be non-negative, got {n_swaps}")

        self.n_swaps = n_swaps
        self._rng = np.random.default_rng()

    def __call__(
        self,
        words: Union[List[str], np.ndarray],
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Apply random swap.

        Args:
            words: List of words
            seed: Random seed

        Returns:
            Words with some swapped
        """
        if isinstance(words, np.ndarray):
            words = words.tolist()

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if len(words) < 2:
            return words

        result = list(words)

        for _ in range(self.n_swaps):
            if len(result) < 2:
                break

            # Random adjacent pair
            idx = self._rng.integers(0, len(result) - 1)
            result[idx], result[idx + 1] = result[idx + 1], result[idx]

        return result

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Composite Text Augmentation
# =============================================================================


class TextAugmenter:
    """
    Composite text augmentation pipeline.

    Combines multiple text augmentations with configurable probabilities.

    Args:
        augmentations: List of (augmentation, probability) tuples
        always_apply: Whether to always apply at least one augmentation

    Example:
        >>> augmenter = TextAugmenter([
        ...     (RandomTokenDeletion(0.1), 0.3),
        ...     (RandomSwap(1), 0.3),
        ...     (RandomTokenReplacement(0.1, vocab_size=10000), 0.2),
        ... ])
        >>> tokens = np.array([1, 2, 3, 4, 5])
        >>> augmented = augmenter(tokens)
    """

    def __init__(
        self,
        augmentations: List[Tuple[Callable, float]],
        always_apply: bool = False,
    ):
        self.augmentations = augmentations
        self.always_apply = always_apply
        self._rng = np.random.default_rng()

    def __call__(
        self,
        data: Union[np.ndarray, List[str]],
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, List[str]]:
        """
        Apply augmentations.

        Args:
            data: Input data (tokens or words)
            seed: Random seed

        Returns:
            Augmented data
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        result = data
        applied = False

        for aug, prob in self.augmentations:
            if self._rng.random() < prob:
                if hasattr(aug, "set_rng"):
                    aug.set_rng(self._rng)
                result = aug(result)
                applied = True

        # If always_apply and nothing was applied, apply first augmentation
        if self.always_apply and not applied and self.augmentations:
            aug, _ = self.augmentations[0]
            if hasattr(aug, "set_rng"):
                aug.set_rng(self._rng)
            result = aug(result)

        return result

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self._rng = rng


# =============================================================================
# Registry
# =============================================================================


TEXT_AUGMENTATIONS: Dict[str, type] = {
    "token_masking": TokenMasking,
    "random_token_masking": RandomTokenMasking,
    "random_token_replacement": RandomTokenReplacement,
    "random_token_deletion": RandomTokenDeletion,
    "random_token_insertion": RandomTokenInsertion,
    "synonym_replacement": SynonymReplacement,
    "random_swap": RandomSwap,
    "text_augmenter": TextAugmenter,
}


def get_text_augmentation(name: str, **kwargs) -> Callable:
    """
    Get text augmentation by name.

    Args:
        name: Augmentation name (case-insensitive)
        **kwargs: Arguments to pass to augmentation constructor

    Returns:
        Augmentation instance

    Raises:
        ValueError: If augmentation name is not found

    Example:
        >>> aug = get_text_augmentation("token_masking", mask_prob=0.15)
        >>> isinstance(aug, TokenMasking)
        True
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower not in TEXT_AUGMENTATIONS:
        available = list(TEXT_AUGMENTATIONS.keys())
        raise ValueError(
            f"Unknown text augmentation '{name}'. Available: {available}"
        )

    return TEXT_AUGMENTATIONS[name_lower](**kwargs)


def list_text_augmentations() -> List[str]:
    """List all available text augmentations."""
    return list(TEXT_AUGMENTATIONS.keys())
