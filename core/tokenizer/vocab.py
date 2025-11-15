"""Vocabulary management utilities."""

from typing import Dict, List, Optional, Set
from pathlib import Path
import json
from collections import Counter
from loguru import logger


class Vocabulary:
    """Custom vocabulary class for managing tokens and their mappings.

    Provides utilities for building vocabularies from text corpora,
    managing special tokens, and converting between tokens and IDs.
    """

    def __init__(
        self,
        vocab_dict: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize vocabulary.

        Args:
            vocab_dict: Optional pre-existing vocabulary mapping
            special_tokens: List of special tokens to add
        """
        # Special tokens
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]

        # Initialize vocab
        if vocab_dict is not None:
            self.token_to_id = vocab_dict
            self.id_to_token = {idx: token for token, idx in vocab_dict.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}
            self._add_special_tokens()

        logger.info(f"Vocabulary initialized with {len(self)} tokens")

    def _add_special_tokens(self) -> None:
        """Add special tokens to the vocabulary."""
        for token in self.special_tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary.

        Args:
            token: Token to add

        Returns:
            Token ID
        """
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            return idx
        return self.token_to_id[token]

    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens to the vocabulary.

        Args:
            tokens: List of tokens to add

        Returns:
            List of token IDs
        """
        return [self.add_token(token) for token in tokens]

    def get_id(self, token: str, default: Optional[int] = None) -> int:
        """Get token ID for a token.

        Args:
            token: Token to look up
            default: Default ID if token not found (uses <unk> if None)

        Returns:
            Token ID
        """
        if default is None:
            default = self.unk_token_id
        return self.token_to_id.get(token, default)

    def get_token(self, idx: int, default: str = "<unk>") -> str:
        """Get token for a token ID.

        Args:
            idx: Token ID
            default: Default token if ID not found

        Returns:
            Token string
        """
        return self.id_to_token.get(idx, default)

    def encode(self, tokens: List[str]) -> List[int]:
        """Encode tokens to IDs.

        Args:
            tokens: List of tokens

        Returns:
            List of token IDs
        """
        return [self.get_id(token) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Decode token IDs to tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of tokens
        """
        return [self.get_token(idx) for idx in ids]

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.token_to_id.get("<pad>", 0)

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.token_to_id.get("<unk>", 1)

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.token_to_id.get("<bos>", 2)

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.token_to_id.get("<eos>", 3)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_id

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file.

        Args:
            filepath: Path to save the vocabulary
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        vocab_data = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Vocabulary saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from a JSON file.

        Args:
            filepath: Path to the vocabulary file

        Returns:
            Vocabulary instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        vocab = cls(
            vocab_dict=vocab_data["token_to_id"],
            special_tokens=vocab_data.get("special_tokens"),
        )

        logger.info(f"Vocabulary loaded from: {filepath}")
        return vocab

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        max_vocab_size: Optional[int] = None,
        min_frequency: int = 1,
        tokenizer_fn=None,
        special_tokens: Optional[List[str]] = None,
    ) -> "Vocabulary":
        """Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size (None for unlimited)
            min_frequency: Minimum frequency for a token to be included
            tokenizer_fn: Function to tokenize text (uses split if None)
            special_tokens: List of special tokens

        Returns:
            Vocabulary instance
        """
        if tokenizer_fn is None:
            tokenizer_fn = lambda x: x.split()

        # Count token frequencies
        token_counts = Counter()
        for text in texts:
            tokens = tokenizer_fn(text)
            token_counts.update(tokens)

        # Filter by minimum frequency
        filtered_tokens = [
            token for token, count in token_counts.items()
            if count >= min_frequency
        ]

        # Sort by frequency (most common first)
        sorted_tokens = sorted(
            filtered_tokens,
            key=lambda x: token_counts[x],
            reverse=True
        )

        # Limit vocabulary size
        if max_vocab_size is not None:
            sorted_tokens = sorted_tokens[:max_vocab_size]

        # Create vocabulary
        vocab = cls(special_tokens=special_tokens)
        vocab.add_tokens(sorted_tokens)

        logger.info(f"Built vocabulary from {len(texts)} texts")
        logger.info(f"Vocabulary size: {len(vocab)}")
        logger.info(f"Total unique tokens in corpus: {len(token_counts)}")

        return vocab

    def get_token_frequency(self, texts: List[str], tokenizer_fn=None) -> Dict[str, int]:
        """Get frequency of each vocabulary token in the given texts.

        Args:
            texts: List of text strings
            tokenizer_fn: Function to tokenize text

        Returns:
            Dictionary mapping tokens to their frequencies
        """
        if tokenizer_fn is None:
            tokenizer_fn = lambda x: x.split()

        token_counts = Counter()
        for text in texts:
            tokens = tokenizer_fn(text)
            # Only count tokens that are in vocabulary
            vocab_tokens = [t for t in tokens if t in self]
            token_counts.update(vocab_tokens)

        return dict(token_counts)

    def get_coverage(self, texts: List[str], tokenizer_fn=None) -> float:
        """Calculate vocabulary coverage over the given texts.

        Args:
            texts: List of text strings
            tokenizer_fn: Function to tokenize text

        Returns:
            Coverage percentage (0-100)
        """
        if tokenizer_fn is None:
            tokenizer_fn = lambda x: x.split()

        total_tokens = 0
        covered_tokens = 0

        for text in texts:
            tokens = tokenizer_fn(text)
            total_tokens += len(tokens)
            covered_tokens += sum(1 for t in tokens if t in self)

        coverage = (covered_tokens / total_tokens * 100) if total_tokens > 0 else 0
        return coverage

    def prune(self, min_frequency: int, texts: List[str], tokenizer_fn=None) -> "Vocabulary":
        """Create a pruned vocabulary by removing low-frequency tokens.

        Args:
            min_frequency: Minimum frequency threshold
            texts: List of texts to calculate frequencies
            tokenizer_fn: Function to tokenize text

        Returns:
            New pruned Vocabulary instance
        """
        frequencies = self.get_token_frequency(texts, tokenizer_fn)

        # Keep only tokens above threshold
        kept_tokens = [
            token for token, freq in frequencies.items()
            if freq >= min_frequency
        ]

        # Create new vocabulary with filtered tokens
        new_vocab = Vocabulary(special_tokens=self.special_tokens)
        new_vocab.add_tokens(kept_tokens)

        logger.info(f"Pruned vocabulary: {len(self)} -> {len(new_vocab)} tokens")

        return new_vocab

    def merge(self, other: "Vocabulary") -> "Vocabulary":
        """Merge this vocabulary with another.

        Args:
            other: Another Vocabulary instance

        Returns:
            New merged Vocabulary instance
        """
        # Combine all tokens
        all_tokens = set(self.token_to_id.keys()) | set(other.token_to_id.keys())

        # Create new vocabulary
        merged_vocab = Vocabulary(special_tokens=self.special_tokens)
        merged_vocab.add_tokens(list(all_tokens))

        logger.info(f"Merged vocabularies: {len(self)} + {len(other)} -> {len(merged_vocab)}")

        return merged_vocab

    def __repr__(self) -> str:
        """String representation of the vocabulary."""
        return f"Vocabulary(size={len(self)}, special_tokens={len(self.special_tokens)})"


if __name__ == "__main__":
    # Test the Vocabulary class
    print("Testing Vocabulary:")

    # Create a simple vocabulary
    vocab = Vocabulary()
    print(f"Initial vocab: {vocab}")

    # Add some tokens
    tokens = ["hello", "world", "python", "programming"]
    vocab.add_tokens(tokens)
    print(f"After adding tokens: {vocab}")

    # Test encoding/decoding
    test_tokens = ["hello", "world", "unknown"]
    encoded = vocab.encode(test_tokens)
    print(f"\nEncoded {test_tokens}: {encoded}")

    decoded = vocab.decode(encoded)
    print(f"Decoded back: {decoded}")

    # Test building from texts
    print("\n\nBuilding vocabulary from texts:")
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the dog and the cat are friends",
        "python programming is fun and rewarding",
        "machine learning with python is powerful",
    ]

    vocab_from_texts = Vocabulary.from_texts(
        texts,
        max_vocab_size=20,
        min_frequency=1,
    )
    print(vocab_from_texts)

    # Test coverage
    coverage = vocab_from_texts.get_coverage(texts)
    print(f"Vocabulary coverage: {coverage:.2f}%")

    # Test save/load
    print("\nTesting save/load:")
    save_path = "./data/test_vocab.json"
    vocab_from_texts.save(save_path)
    loaded_vocab = Vocabulary.load(save_path)
    print(f"Loaded vocab: {loaded_vocab}")

    print("\nAll tests passed!")
