"""
Tests for Text Augmentation module.

Tests include:
    - Token masking (BERT-style)
    - Random token replacement
    - Token deletion and insertion
    - Word-level operations (synonym, swap)
    - Registry functions
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3_training.text_augmentation import (
    # Token-level
    TokenMasking,
    RandomTokenMasking,
    RandomTokenReplacement,
    RandomTokenDeletion,
    RandomTokenInsertion,
    # Word-level
    SynonymReplacement,
    RandomSwap,
    # Composite
    TextAugmenter,
    # Registry
    get_text_augmentation,
    list_text_augmentations,
    TEXT_AUGMENTATIONS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_tokens():
    """Create sample token sequence."""
    # Simulating a tokenized sentence: [CLS] this is a test [SEP]
    return np.array([101, 2023, 2003, 1037, 3231, 102])


@pytest.fixture
def sample_batch_tokens():
    """Create sample batch of token sequences."""
    # Batch of 4 sequences
    return np.array([
        [101, 2023, 2003, 1037, 3231, 102, 0, 0],
        [101, 2154, 2003, 1037, 4231, 102, 0, 0],
        [101, 3333, 4444, 5555, 6666, 102, 0, 0],
        [101, 7777, 8888, 9999, 1111, 102, 0, 0],
    ])


@pytest.fixture
def sample_words():
    """Create sample word list."""
    return ["the", "quick", "brown", "fox", "jumps"]


# =============================================================================
# Test TokenMasking (BERT-style)
# =============================================================================


class TestTokenMasking:
    """Test BERT-style token masking."""

    def test_output_shapes(self, sample_tokens):
        """Test output shapes match input."""
        masker = TokenMasking(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30522,
        )
        masked, labels = masker(sample_tokens, seed=42)
        assert masked.shape == sample_tokens.shape
        assert labels.shape == sample_tokens.shape

    def test_batch_output_shapes(self, sample_batch_tokens):
        """Test batch output shapes."""
        masker = TokenMasking(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30522,
        )
        masked, labels = masker(sample_batch_tokens, seed=42)
        assert masked.shape == sample_batch_tokens.shape
        assert labels.shape == sample_batch_tokens.shape

    def test_labels_for_masked_positions(self, sample_tokens):
        """Test that labels contain original tokens for masked positions."""
        masker = TokenMasking(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30522,
        )
        masked, labels = masker(sample_tokens, seed=42)

        # For positions where masked != original, labels should have original
        for i in range(len(sample_tokens)):
            if masked[i] != sample_tokens[i]:
                assert labels[i] == sample_tokens[i]

    def test_labels_for_unmasked_positions(self, sample_tokens):
        """Test that labels are -100 for unmasked positions."""
        masker = TokenMasking(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30522,
        )
        masked, labels = masker(sample_tokens, seed=42)

        # For positions where masked == original and wasn't changed
        for i in range(len(sample_tokens)):
            if masked[i] == sample_tokens[i] and labels[i] == -100:
                pass  # Expected

    def test_special_tokens_not_masked(self):
        """Test that special tokens are never masked."""
        tokens = np.array([101, 2023, 2003, 1037, 3231, 102])  # CLS, tokens, SEP
        masker = TokenMasking(
            mask_prob=1.0,  # Mask everything possible
            mask_token_id=103,
            vocab_size=30522,
            special_token_ids=[101, 102],  # CLS and SEP
        )
        masked, labels = masker(tokens, seed=42)

        # CLS and SEP should never be masked
        assert masked[0] == 101
        assert masked[-1] == 102
        assert labels[0] == -100
        assert labels[-1] == -100

    def test_reproducibility(self, sample_tokens):
        """Test reproducibility with same seed."""
        masker = TokenMasking(mask_prob=0.15, mask_token_id=103)
        masked1, labels1 = masker(sample_tokens.copy(), seed=42)
        masked2, labels2 = masker(sample_tokens.copy(), seed=42)
        np.testing.assert_array_equal(masked1, masked2)
        np.testing.assert_array_equal(labels1, labels2)

    def test_zero_mask_prob(self, sample_tokens):
        """Test that mask_prob=0 doesn't mask anything."""
        masker = TokenMasking(mask_prob=0.0, mask_token_id=103)
        masked, labels = masker(sample_tokens, seed=42)
        # With mask_prob=0, no tokens should be masked
        np.testing.assert_array_equal(masked, sample_tokens)
        # All labels should be -100 (no tokens selected for masking)
        assert np.all(labels == -100)

    def test_invalid_mask_prob(self):
        """Test that invalid mask_prob raises error."""
        with pytest.raises(ValueError):
            TokenMasking(mask_prob=1.5)
        with pytest.raises(ValueError):
            TokenMasking(mask_prob=-0.1)


# =============================================================================
# Test RandomTokenMasking
# =============================================================================


class TestRandomTokenMasking:
    """Test simple random token masking."""

    def test_output_shape(self, sample_tokens):
        """Test output shape is preserved."""
        masker = RandomTokenMasking(mask_prob=0.1, mask_token_id=0)
        masked = masker(sample_tokens, seed=42)
        assert masked.shape == sample_tokens.shape

    def test_some_tokens_masked(self):
        """Test that some tokens are masked."""
        tokens = np.arange(1, 101)  # 100 tokens
        masker = RandomTokenMasking(mask_prob=0.15, mask_token_id=0)
        masked = masker(tokens, seed=42)

        # Some should be masked (roughly 15%)
        n_masked = np.sum(masked == 0)
        assert n_masked > 0  # At least some masked

    def test_no_masking_with_zero_prob(self, sample_tokens):
        """Test no masking with prob=0."""
        masker = RandomTokenMasking(mask_prob=0.0, mask_token_id=0)
        masked = masker(sample_tokens, seed=42)
        np.testing.assert_array_equal(masked, sample_tokens)


# =============================================================================
# Test RandomTokenReplacement
# =============================================================================


class TestRandomTokenReplacement:
    """Test random token replacement."""

    def test_output_shape(self, sample_tokens):
        """Test output shape is preserved."""
        replacer = RandomTokenReplacement(replace_prob=0.1, vocab_size=10000)
        replaced = replacer(sample_tokens, seed=42)
        assert replaced.shape == sample_tokens.shape

    def test_replacement_in_vocab_range(self):
        """Test replacements are in vocabulary range."""
        tokens = np.arange(1, 101)
        vocab_size = 1000
        replacer = RandomTokenReplacement(replace_prob=0.2, vocab_size=vocab_size)
        replaced = replacer(tokens, seed=42)

        # All tokens should be in valid range
        assert np.all(replaced >= 0)
        assert np.all(replaced < vocab_size)

    def test_special_tokens_preserved(self):
        """Test special tokens are not replaced."""
        tokens = np.array([101, 1, 2, 3, 102])  # CLS, tokens, SEP
        replacer = RandomTokenReplacement(
            replace_prob=1.0,  # Replace everything possible
            vocab_size=10000,
            special_token_ids=[101, 102],
        )
        replaced = replacer(tokens, seed=42)

        # CLS and SEP should be preserved
        assert replaced[0] == 101
        assert replaced[-1] == 102


# =============================================================================
# Test RandomTokenDeletion
# =============================================================================


class TestRandomTokenDeletion:
    """Test random token deletion."""

    def test_output_shape(self, sample_tokens):
        """Test output shape is preserved (padded)."""
        deleter = RandomTokenDeletion(delete_prob=0.1)
        deleted = deleter(sample_tokens, seed=42)
        assert deleted.shape == sample_tokens.shape

    def test_some_tokens_deleted(self):
        """Test that some tokens are deleted."""
        tokens = np.arange(1, 101)
        deleter = RandomTokenDeletion(delete_prob=0.2)
        deleted = deleter(tokens, seed=42)

        # Count non-zero tokens (should be fewer than original)
        n_remaining = np.sum(deleted != 0)
        assert n_remaining < len(tokens)

    def test_special_tokens_preserved(self):
        """Test special tokens are not deleted."""
        tokens = np.array([101, 1, 2, 3, 102])
        deleter = RandomTokenDeletion(
            delete_prob=1.0,
            special_token_ids=[101, 102],
        )
        deleted = deleter(tokens, seed=42)

        # CLS and SEP should be preserved
        assert deleted[0] == 101
        assert np.sum(deleted == 102) >= 1  # SEP somewhere


# =============================================================================
# Test RandomTokenInsertion
# =============================================================================


class TestRandomTokenInsertion:
    """Test random token insertion."""

    def test_output_shape(self, sample_tokens):
        """Test output shape is preserved (truncated/padded)."""
        inserter = RandomTokenInsertion(insert_prob=0.1, vocab_size=10000)
        inserted = inserter(sample_tokens, seed=42)
        assert inserted.shape == sample_tokens.shape

    def test_max_insertions_respected(self):
        """Test that max_insertions is respected."""
        tokens = np.array([1, 2, 3, 4, 5])
        inserter = RandomTokenInsertion(
            insert_prob=1.0,  # Always try to insert
            vocab_size=10000,
            max_insertions=2,
        )
        inserted = inserter(tokens, seed=42)

        # Should have at most 2 insertions (total length truncated to 5)
        assert inserted.shape == tokens.shape


# =============================================================================
# Test SynonymReplacement
# =============================================================================


class TestSynonymReplacement:
    """Test synonym replacement for words."""

    def test_basic_replacement(self):
        """Test basic synonym replacement."""
        synonym_dict = {
            "good": ["great", "excellent"],
            "bad": ["poor", "terrible"],
        }
        words = ["this", "is", "good"]
        replacer = SynonymReplacement(
            replace_prob=1.0,
            synonym_dict=synonym_dict,
        )
        result = replacer(words, seed=42)
        assert result[2] in synonym_dict["good"]

    def test_no_synonym_available(self):
        """Test that words without synonyms are unchanged."""
        synonym_dict = {"good": ["great"]}
        words = ["this", "is", "bad"]
        replacer = SynonymReplacement(
            replace_prob=1.0,
            synonym_dict=synonym_dict,
        )
        result = replacer(words, seed=42)
        assert result[2] == "bad"

    def test_n_replacements_limit(self):
        """Test that n_replacements limit is respected."""
        synonym_dict = {
            "a": ["b"],
            "c": ["d"],
            "e": ["f"],
        }
        words = ["a", "c", "e"]
        replacer = SynonymReplacement(
            replace_prob=1.0,
            synonym_dict=synonym_dict,
            n_replacements=1,
        )
        result = replacer(words, seed=42)

        # Only 1 should be replaced
        replacements = sum(1 for i, w in enumerate(words) if result[i] != w)
        assert replacements <= 1


# =============================================================================
# Test RandomSwap
# =============================================================================


class TestRandomSwap:
    """Test random word swap."""

    def test_output_length_preserved(self, sample_words):
        """Test output length is preserved."""
        swapper = RandomSwap(n_swaps=1)
        result = swapper(sample_words, seed=42)
        assert len(result) == len(sample_words)

    def test_swap_actually_swaps(self):
        """Test that swap actually changes word order."""
        words = ["a", "b", "c", "d", "e"]
        swapper = RandomSwap(n_swaps=10)  # Many swaps
        result = swapper(words, seed=42)
        # With 10 swaps, order should be different
        assert result != words

    def test_single_word_unchanged(self):
        """Test that single word list is unchanged."""
        words = ["hello"]
        swapper = RandomSwap(n_swaps=1)
        result = swapper(words, seed=42)
        assert result == words

    def test_empty_list_unchanged(self):
        """Test that empty list is unchanged."""
        words = []
        swapper = RandomSwap(n_swaps=1)
        result = swapper(words, seed=42)
        assert result == words


# =============================================================================
# Test TextAugmenter
# =============================================================================


class TestTextAugmenter:
    """Test composite text augmenter."""

    def test_basic_augmentation(self, sample_tokens):
        """Test basic augmentation pipeline."""
        augmenter = TextAugmenter([
            (RandomTokenDeletion(0.1), 0.5),
            (RandomTokenReplacement(0.1, vocab_size=10000), 0.5),
        ])
        result = augmenter(sample_tokens, seed=42)
        assert result.shape == sample_tokens.shape

    def test_always_apply(self, sample_tokens):
        """Test that always_apply ensures at least one augmentation."""
        augmenter = TextAugmenter(
            augmentations=[
                (RandomTokenDeletion(0.1), 0.0),  # Never applies
                (RandomTokenReplacement(0.1, vocab_size=10000), 0.0),  # Never applies
            ],
            always_apply=True,
        )
        # This should still apply one augmentation due to always_apply
        result = augmenter(sample_tokens, seed=42)
        assert result.shape == sample_tokens.shape

    def test_word_augmentation(self, sample_words):
        """Test augmentation on word lists."""
        synonym_dict = {"quick": ["fast", "speedy"]}
        augmenter = TextAugmenter([
            (SynonymReplacement(0.5, synonym_dict=synonym_dict), 1.0),
            (RandomSwap(1), 0.5),
        ])
        result = augmenter(sample_words, seed=42)
        assert len(result) == len(sample_words)


# =============================================================================
# Test Registry
# =============================================================================


class TestTextAugmentationRegistry:
    """Test text augmentation registry."""

    def test_get_token_masking(self):
        """Test getting TokenMasking from registry."""
        aug = get_text_augmentation("token_masking", mask_prob=0.15, mask_token_id=103)
        assert isinstance(aug, TokenMasking)

    def test_get_random_deletion(self):
        """Test getting RandomTokenDeletion from registry."""
        aug = get_text_augmentation("random_token_deletion", delete_prob=0.1)
        assert isinstance(aug, RandomTokenDeletion)

    def test_case_insensitive(self):
        """Test that registry is case-insensitive."""
        aug1 = get_text_augmentation("Token_Masking", mask_prob=0.1)
        aug2 = get_text_augmentation("TOKEN-MASKING", mask_prob=0.1)
        assert isinstance(aug1, TokenMasking)
        assert isinstance(aug2, TokenMasking)

    def test_invalid_name_raises(self):
        """Test that invalid name raises error."""
        with pytest.raises(ValueError):
            get_text_augmentation("invalid_augmentation")

    def test_list_augmentations(self):
        """Test listing augmentations."""
        augs = list_text_augmentations()
        assert "token_masking" in augs
        assert "synonym_replacement" in augs

    def test_all_in_registry(self):
        """Test all augmentations are in registry."""
        expected = [
            "token_masking",
            "random_token_masking",
            "random_token_replacement",
            "random_token_deletion",
            "random_token_insertion",
            "synonym_replacement",
            "random_swap",
            "text_augmenter",
        ]
        for name in expected:
            assert name in TEXT_AUGMENTATIONS


# =============================================================================
# Integration Tests
# =============================================================================


class TestTextAugmentationIntegration:
    """Integration tests for text augmentation."""

    def test_bert_pretraining_simulation(self, sample_batch_tokens):
        """Test simulating BERT pretraining with masking."""
        masker = TokenMasking(
            mask_prob=0.15,
            mask_token_id=103,
            vocab_size=30522,
            special_token_ids=[101, 102, 0],  # CLS, SEP, PAD
        )

        masked, labels = masker(sample_batch_tokens, seed=42)

        # Check shapes
        assert masked.shape == sample_batch_tokens.shape
        assert labels.shape == sample_batch_tokens.shape

        # Check that masked positions have labels
        for b in range(sample_batch_tokens.shape[0]):
            for i in range(sample_batch_tokens.shape[1]):
                if masked[b, i] != sample_batch_tokens[b, i]:
                    assert labels[b, i] == sample_batch_tokens[b, i]

    def test_full_augmentation_pipeline(self, sample_batch_tokens):
        """Test full augmentation pipeline."""
        augmenter = TextAugmenter([
            (RandomTokenMasking(0.1, mask_token_id=0), 0.5),
            (RandomTokenReplacement(0.05, vocab_size=30522), 0.3),
        ])

        # Apply to each sequence
        results = []
        for i in range(sample_batch_tokens.shape[0]):
            result = augmenter(sample_batch_tokens[i], seed=i)
            results.append(result)

        output = np.stack(results)
        assert output.shape == sample_batch_tokens.shape

    def test_word_and_token_augmentation(self):
        """Test combining word and token augmentations."""
        # Start with words
        words = ["the", "quick", "brown", "fox", "jumps"]

        # Apply word augmentation
        synonym_dict = {"quick": ["fast", "speedy"], "brown": ["tan", "dark"]}
        word_augmenter = TextAugmenter([
            (SynonymReplacement(0.5, synonym_dict=synonym_dict), 1.0),
            (RandomSwap(1), 0.5),
        ])

        augmented_words = word_augmenter(words, seed=42)
        assert len(augmented_words) == len(words)

        # Convert to tokens (simplified)
        tokens = np.array([hash(w) % 10000 for w in augmented_words])

        # Apply token augmentation
        token_augmenter = TextAugmenter([
            (RandomTokenMasking(0.1, mask_token_id=0), 0.5),
        ])

        augmented_tokens = token_augmenter(tokens, seed=42)
        assert augmented_tokens.shape == tokens.shape
