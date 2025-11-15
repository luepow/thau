"""Tokenizer implementation for text processing."""

import torch
from typing import List, Optional, Union, Dict, Any
from transformers import AutoTokenizer
from loguru import logger
from pathlib import Path


class Tokenizer:
    """Wrapper around HuggingFace tokenizers with additional utilities.

    Provides a unified interface for tokenization with support for:
    - Loading pretrained tokenizers
    - Training custom tokenizers
    - Batch encoding/decoding
    - Special tokens handling
    """

    def __init__(
        self,
        tokenizer_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_length: int = 2048,
        padding_side: str = "right",
        truncation: bool = True,
    ):
        """Initialize the tokenizer.

        Args:
            tokenizer_name_or_path: Name or path of the tokenizer
            max_length: Maximum sequence length
            padding_side: Side to pad sequences ("left" or "right")
            truncation: Whether to truncate sequences
        """
        self.tokenizer_name = tokenizer_name_or_path
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation = truncation

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        logger.info(f"Tokenizer loaded: {tokenizer_name_or_path}")
        logger.info(f"Vocab size: {self.vocab_size}")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer from HuggingFace or local path.

        Returns:
            Loaded AutoTokenizer instance
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
            )

            # Configure tokenizer
            tokenizer.padding_side = self.padding_side
            tokenizer.truncation_side = "right"
            tokenizer.model_max_length = self.max_length

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logger.info(f"Set pad_token to: {tokenizer.pad_token}")

            return tokenizer

        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        return_tensors: Optional[str] = "pt",
        return_attention_mask: bool = True,
    ) -> Union[List[int], Dict[str, torch.Tensor]]:
        """Encode text to token IDs.

        Args:
            text: Text or list of texts to encode
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            padding: Whether to pad sequences
            return_tensors: Format of returned tensors ("pt" for PyTorch, None for list)
            return_attention_mask: Whether to return attention mask

        Returns:
            Dictionary with input_ids and optionally attention_mask
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

        return encoded

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        return text

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs to texts.

        Args:
            token_ids: Batch of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces

        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        texts = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        return texts

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens (not IDs).

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to their IDs.

        Args:
            tokens: List of tokens

        Returns:
            List of token IDs
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of tokens
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_special_tokens(self) -> Dict[str, Any]:
        """Get all special tokens and their IDs.

        Returns:
            Dictionary of special tokens
        """
        return {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
            "sep_token": self.tokenizer.sep_token,
            "cls_token": self.tokenizer.cls_token,
            "mask_token": self.tokenizer.mask_token,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        """Add special tokens to the tokenizer.

        Args:
            special_tokens_dict: Dictionary of special tokens to add

        Returns:
            Number of tokens added
        """
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added} special tokens")
        return num_added

    def save(self, save_directory: str) -> None:
        """Save the tokenizer to a directory.

        Args:
            save_directory: Directory to save the tokenizer
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Tokenizer saved to: {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_name_or_path: str,
        **kwargs
    ) -> "Tokenizer":
        """Load a pretrained tokenizer.

        Args:
            tokenizer_name_or_path: Name or path of the tokenizer
            **kwargs: Additional arguments

        Returns:
            Tokenizer instance
        """
        return cls(tokenizer_name_or_path=tokenizer_name_or_path, **kwargs)

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        return self.tokenizer.get_vocab()

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Make the tokenizer callable for encoding.

        Args:
            text: Text or list of texts to encode
            **kwargs: Additional arguments for encoding

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.encode(text, **kwargs)

    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        return (
            f"Tokenizer(\n"
            f"  name={self.tokenizer_name},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_length={self.max_length},\n"
            f"  padding_side={self.padding_side}\n"
            f")"
        )


class ConversationTokenizer(Tokenizer):
    """Specialized tokenizer for chat/conversation formatting.

    Handles conversation templates for instruct/chat models.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """Initialize conversation tokenizer.

        Args:
            tokenizer_name_or_path: Name or path of the tokenizer
            max_length: Maximum sequence length
            system_prompt: Optional system prompt for conversations
        """
        super().__init__(tokenizer_name_or_path, max_length)
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a conversation using the chat template.

        Args:
            messages: List of message dicts with "role" and "content" keys
            add_generation_prompt: Whether to add prompt for generation

        Returns:
            Formatted conversation string
        """
        # Check if tokenizer has chat template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Fallback to simple formatting
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    formatted += f"<|assistant|>\n{content}\n"

            if add_generation_prompt:
                formatted += "<|assistant|>\n"

        return formatted

    def encode_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encode a conversation to token IDs.

        Args:
            messages: List of message dicts
            add_generation_prompt: Whether to add prompt for generation
            **kwargs: Additional encoding arguments

        Returns:
            Dictionary with input_ids and attention_mask
        """
        formatted_text = self.format_conversation(messages, add_generation_prompt)
        return self.encode(formatted_text, **kwargs)


if __name__ == "__main__":
    # Test the tokenizer
    print("Testing Tokenizer:")
    tokenizer = Tokenizer()
    print(tokenizer)

    # Test encoding
    text = "Hello, how are you today?"
    encoded = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Encoded shape: {encoded['input_ids'].shape}")
    print(f"Token IDs: {encoded['input_ids'][0].tolist()}")

    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"Decoded text: {decoded}")

    # Test batch encoding/decoding
    texts = ["Hello!", "How are you?", "Nice to meet you."]
    batch_encoded = tokenizer.encode(texts)
    print(f"\nBatch encoded shape: {batch_encoded['input_ids'].shape}")

    batch_decoded = tokenizer.batch_decode(batch_encoded['input_ids'])
    print(f"Batch decoded: {batch_decoded}")

    # Test special tokens
    print("\nSpecial tokens:")
    import json
    print(json.dumps(tokenizer.get_special_tokens(), indent=2, default=str))

    # Test conversation tokenizer
    print("\n\nTesting ConversationTokenizer:")
    conv_tokenizer = ConversationTokenizer()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Spain?"},
    ]

    formatted = conv_tokenizer.format_conversation(messages)
    print(f"Formatted conversation:\n{formatted}")

    print("\nAll tests passed!")
