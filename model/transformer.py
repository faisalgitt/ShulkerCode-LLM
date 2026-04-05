"""
=====================================
 Shulker Code — Transformer Model
 GPT-style Decoder with SwiGLU FFN
 Developed by @kopeedev / CyeroX
=====================================
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import GroupedQueryAttention


# ─────────────────────────────────────────────
# Model Configuration Dataclass
# ─────────────────────────────────────────────
@dataclass
class ShulkerConfig:
    """Complete configuration for a Shulker Code model."""

    # Identity
    name: str = "shulker-nano"
    version: str = "1.0.0"

    # Vocabulary
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Architecture
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4          # Set < num_heads for GQA
    intermediate_size: int = 1024  # FFN hidden size
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_embeddings: bool = True    # Tie input/output embeddings

    # Normalization
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShulkerConfig":
        """Load config from a dict (e.g. parsed from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict."""
        import dataclasses
        return dataclasses.asdict(self)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        params = self.vocab_size * self.hidden_size
        # Each layer: attention + FFN + norms
        layer_params = (
            # Attention projections
            self.hidden_size * self.hidden_size +          # Q
            self.hidden_size * (self.num_kv_heads * self.head_dim) * 2 +  # K + V
            self.hidden_size * self.hidden_size +          # O
            # FFN (SwiGLU has 3 matrices)
            self.hidden_size * self.intermediate_size * 3 +
            # RMS Norms (2 per layer)
            self.hidden_size * 2
        )
        params += layer_params * self.num_layers
        # Final norm + LM head
        params += self.hidden_size + (0 if self.tie_embeddings else self.vocab_size * self.hidden_size)
        return params


# ─────────────────────────────────────────────
# RMS Layer Normalization
# Simpler and faster than LayerNorm
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Omits the mean-centering for efficiency.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ─────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# Used in LLaMA, PaLM, etc. — better than GELU FFN
# ─────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    FFN(x) = (Swish(xW1) ⊙ xW3) W2
    Three weight matrices instead of two.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # W1
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)  # W3
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # W2
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * swish activation
        gate = F.silu(self.gate_proj(x))   # Swish(xW1)
        up = self.up_proj(x)               # xW3
        return self.down_proj(self.dropout(gate * up))


# ─────────────────────────────────────────────
# Transformer Decoder Layer
# Pre-normalization (like LLaMA)
# ─────────────────────────────────────────────
class ShulkerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    Uses pre-normalization (norm before attention/FFN).
    Architecture: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """
    def __init__(self, config: ShulkerConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # Self-attention (with GQA + RoPE)
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
        )

        # Pre-normalization
        self.input_layernorm   = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass of one decoder layer.
        Returns updated hidden states and optional KV cache.
        """
        # Pre-norm → Attention → Residual
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(
            x_norm, attention_mask=attention_mask,
            past_kv=past_kv, use_cache=use_cache
        )
        x = residual + attn_out

        # Pre-norm → FFN → Residual
        residual = x
        x = residual + self.ffn(self.post_attn_layernorm(x))

        return x, new_kv


# ─────────────────────────────────────────────
# Shulker Code — Main Model
# ─────────────────────────────────────────────
class ShulkerCodeModel(nn.Module):
    """
    Shulker Code: A GPT-style autoregressive language model
    optimized for code generation tasks.

    Features:
    - GQA (Grouped Query Attention) for memory efficiency
    - RoPE positional embeddings for long context
    - SwiGLU FFN for better performance
    - RMSNorm for stability
    - KV caching for fast inference
    - Configurable size: 5M to 80B+ parameters

    Developed by @kopeedev / CyeroX Development
    """

    def __init__(self, config: ShulkerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            ShulkerDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Language model head (projects to vocab)
        if config.tie_embeddings:
            # Reuse embedding weights — reduces params, improves generalization
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT conventions."""
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count actual model parameters."""
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass of the Shulker Code model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            past_key_values: KV cache from previous forward passes
            use_cache: Whether to cache K/V tensors
            labels: Target token IDs for computing loss

        Returns:
            dict with keys: 'logits', 'loss' (if labels given), 'past_key_values' (if caching)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Initialize past KV cache
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers

        # Token embeddings
        x = self.embed_dropout(self.embed_tokens(input_ids))  # [B, T, hidden]

        # Pass through all decoder layers
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            x, new_kv = layer(
                x,
                attention_mask=attention_mask,
                past_kv=past_key_values[i],
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        # Final layer norm
        x = self.norm(x)  # [B, T, hidden]

        # Project to vocabulary
        if self.config.tie_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)  # [B, T, vocab]
        else:
            logits = self.lm_head(x)  # [B, T, vocab]

        # Compute cross-entropy loss if training
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_past_key_values,
            "hidden_states": x,
        }

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "ShulkerCodeModel":
        """Load a pretrained Shulker Code model from a checkpoint."""
        import json, os
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "model.pt")

        with open(config_path) as f:
            config_dict = json.load(f)

        config = ShulkerConfig.from_dict(config_dict)
        model = cls(config)

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Loaded {config.name} ({model.num_parameters():,} params) from {path}")
        return model.to(device)

    def save_pretrained(self, path: str):
        """Save model weights and config to a directory."""
        import json, os
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        print(f"✅ Model saved to {path}")
