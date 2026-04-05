"""
=====================================
 Shulker Code — Inference Engine
 Fast code generation with streaming
 Developed by @kopeedev / CyeroX
=====================================

Features:
 - Greedy / Top-K / Top-P (nucleus) sampling
 - Temperature scaling
 - KV caching for fast autoregressive decoding
 - Streaming token output (typing effect)
 - Batch generation support
"""

import time
from typing import Optional, List, Iterator, Callable

import torch
import torch.nn.functional as F

from model.transformer import ShulkerCodeModel


class GenerationConfig:
    """Parameters that control how the model generates text."""
    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        greedy: bool = False,         # Overrides do_sample
        stop_tokens: Optional[List[int]] = None,
        eos_token_id: Optional[int] = 2,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample and not greedy
        self.greedy = greedy
        self.stop_tokens = stop_tokens or []
        self.eos_token_id = eos_token_id


class ShulkerInferenceEngine:
    """
    Fast inference engine for Shulker Code.

    Implements:
    - KV caching (avoids recomputing attention for past tokens)
    - Top-K and nucleus (Top-P) sampling
    - Repetition penalty
    - Streaming output
    """

    def __init__(
        self,
        model: ShulkerCodeModel,
        tokenizer,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        language: str = "python",
        task: str = "gen",
    ) -> str:
        """
        Generate code from a prompt.

        Args:
            prompt: Natural language or code prompt
            config: Generation hyperparameters
            language: Target programming language
            task: Task type ("gen", "fix", "explain", "optimize")

        Returns:
            Generated code string
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt with code-specific tokens
        input_ids = self.tokenizer.encode_code(
            prompt, language=language, task=task,
            max_length=self.model.config.max_seq_len - config.max_new_tokens,
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate tokens
        output_ids = self._generate_tokens(input_tensor, config)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][len(input_ids):].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        language: str = "python",
        task: str = "gen",
        callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens one by one (typing effect).

        Args:
            prompt: Input prompt
            config: Generation config
            language: Target language
            task: Task type
            callback: Optional function called with each new token string

        Yields:
            Decoded string for each new token
        """
        if config is None:
            config = GenerationConfig()

        input_ids = self.tokenizer.encode_code(
            prompt, language=language, task=task,
            max_length=self.model.config.max_seq_len - config.max_new_tokens,
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        past_key_values = None
        generated_ids = []

        for _ in range(config.max_new_tokens):
            # Use KV cache: only pass NEW token(s) on subsequent steps
            if past_key_values is None:
                curr_input = input_tensor
            else:
                # Only feed the last generated token
                curr_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=self.device)

            # Forward pass
            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :]  # Last token logits

            # Sample next token
            next_token_id = self._sample_token(
                logits,
                generated_ids + input_ids,
                config,
            )

            # Check stopping conditions
            if next_token_id == config.eos_token_id:
                break
            if next_token_id in config.stop_tokens:
                break

            generated_ids.append(next_token_id)

            # Decode and yield this token
            token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            if callback:
                callback(token_str)
            yield token_str

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Run the full autoregressive generation loop with KV caching."""
        past_key_values = None
        generated = input_ids.clone()
        original_len = input_ids.shape[1]
        all_ids = input_ids[0].tolist()

        for step in range(config.max_new_tokens):
            # Feed only new token(s) when using KV cache
            if past_key_values is None:
                curr_input = generated
            else:
                curr_input = generated[:, -1:]

            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :]

            next_token_id = self._sample_token(logits, all_ids, config)

            if next_token_id == config.eos_token_id:
                break
            if next_token_id in config.stop_tokens:
                break

            all_ids.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            generated = torch.cat([generated, next_token_tensor], dim=1)

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        previous_ids: List[int],
        config: GenerationConfig,
    ) -> int:
        """
        Sample the next token from logits.

        Applies:
        1. Temperature scaling
        2. Repetition penalty
        3. Top-K filtering
        4. Top-P (nucleus) filtering
        5. Sampling or greedy decoding
        """
        # Greedy decoding: pick the highest probability token
        if config.greedy or not config.do_sample:
            return logits.argmax(dim=-1).item()

        # Apply temperature (higher = more random, lower = more deterministic)
        if config.temperature != 1.0:
            logits = logits / max(config.temperature, 1e-8)

        # Repetition penalty: reduce probability of already-seen tokens
        if config.repetition_penalty != 1.0 and previous_ids:
            for token_id in set(previous_ids):
                if token_id < logits.shape[-1]:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= config.repetition_penalty
                    else:
                        logits[0, token_id] *= config.repetition_penalty

        # Top-K filtering: keep only the K most likely tokens
        if config.top_k > 0:
            top_k = min(config.top_k, logits.shape[-1])
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_top_k, float("-inf"))

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-P (nucleus) filtering: keep smallest set of tokens with cumulative prob >= top_p
        if config.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens whose cumulative prob exceeds top_p
            sorted_remove = cumulative_probs - sorted_probs > config.top_p
            sorted_probs[sorted_remove] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            # Scatter back to original ordering
            probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

        # Sample from the filtered distribution
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

    def benchmark(self, prompt: str = "def hello():", num_tokens: int = 100) -> dict:
        """Benchmark inference speed (tokens/sec)."""
        config = GenerationConfig(max_new_tokens=num_tokens, greedy=True)

        start = time.perf_counter()
        output = self.generate(prompt, config)
        elapsed = time.perf_counter() - start

        tokens_generated = len(self.tokenizer.encode(output))
        tps = tokens_generated / elapsed

        return {
            "tokens_generated": tokens_generated,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tps,
            "device": str(self.device),
        }
