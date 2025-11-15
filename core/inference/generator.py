"""Text generation utilities."""

import torch
from typing import Optional, List, Dict, Any
from loguru import logger

from adapters.model_adapter import ModelAdapter
from config.base_config import get_config


class TextGenerator:
    """Text generation with various decoding strategies."""

    def __init__(
        self,
        model_adapter: Optional[ModelAdapter] = None,
        config=None,
    ):
        """Initialize text generator.

        Args:
            model_adapter: ModelAdapter instance
            config: Configuration object
        """
        self.config = config or get_config()

        if model_adapter is None:
            self.model_adapter = ModelAdapter(model_name=self.config.MODEL_NAME)
            self.model_adapter.load_model()
            self.model_adapter.load_tokenizer()
        else:
            self.model_adapter = model_adapter

        self.model = self.model_adapter.model
        self.tokenizer = self.model_adapter.tokenizer

        # Set model to eval mode
        self.model.eval()

        logger.info("TextGenerator initialized")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample
            num_return_sequences: Number of sequences to return

        Returns:
            List of generated texts
        """
        # Use config defaults if not provided
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        top_k = top_k or self.config.TOP_K
        top_p = top_p or self.config.TOP_P

        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.MAX_LENGTH,
        )

        inputs = {k: self.model_adapter.device_manager.to_device(v) for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_texts

    def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Chat with the model.

        Args:
            message: User message
            conversation_history: Optional conversation history
            **kwargs: Generation parameters

        Returns:
            Model response
        """
        # Build conversation
        messages = conversation_history or []
        messages.append({"role": "user", "content": message})

        # Format with chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple formatting
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant:"

        # Generate response
        responses = self.generate(prompt, num_return_sequences=1, **kwargs)

        # Extract assistant response
        response = responses[0]

        # Try to extract only the new response
        if prompt in response:
            response = response[len(prompt):].strip()

        return response


class StreamingGenerator(TextGenerator):
    """Generator with streaming support."""

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
    ):
        """Generate text with streaming (yields tokens).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Yields:
            Generated tokens one at a time
        """
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        top_k = top_k or self.config.TOP_K
        top_p = top_p or self.config.TOP_P

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = self.model_adapter.device_manager.to_device(inputs["input_ids"])

        # Generate token by token
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
