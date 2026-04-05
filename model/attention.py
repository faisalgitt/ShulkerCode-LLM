"""
=====================================
 Shulker Code — Attention Module
 Multi-Head Attention + GQA + RoPE
 Developed by @kopeedev / CyeroX
=====================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Rotary Positional Encoding (RoPE)
# Used in modern LLMs (LLaMA, Mistral, etc.)
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding.
    Encodes position by rotating query/key vectors in pairs.
    Naturally handles long contexts better than absolute PE.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache cos/sin for fast inference
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute and cache the rotation matrices."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)        # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        Args:
            q: Query tensor [batch, heads, seq, dim]
            k: Key tensor   [batch, kv_heads, seq, dim]
            seq_len: Current sequence length
            offset: KV cache offset for incremental decoding
        """
        # Extend cache if needed
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)

        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────
# Grouped Query Attention (GQA)
# Efficient variant: fewer KV heads = less VRAM
# ─────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) as used in LLaMA 2, Mistral, etc.

    - num_heads: number of query heads
    - num_kv_heads: number of key/value heads (≤ num_heads)
    - When num_kv_heads == num_heads → standard MHA
    - When num_kv_heads == 1 → Multi-Query Attention (MQA)
    - Otherwise → GQA (balanced efficiency/quality)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.groups = num_heads // num_kv_heads   # How many Q heads share each KV head
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for GQA.
        Args:
            x: Input hidden states [batch, seq, hidden]
            attention_mask: Causal or padding mask
            past_kv: Cached K/V tensors for fast generation
            use_cache: Whether to return updated KV cache
        Returns:
            output: Attended output [batch, seq, hidden]
            new_kv: Updated KV cache (or None)
        """
        B, T, C = x.shape
        offset = past_kv[0].shape[2] if past_kv is not None else 0

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary(q, k, T, offset)

        # Append to KV cache if using incremental decoding
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]

        # Expand KV heads to match Q heads (GQA expansion)
        if self.groups > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.groups, kv_seq_len, self.head_dim)
            k = k.reshape(B, self.num_heads, kv_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.groups, kv_seq_len, self.head_dim)
            v = v.reshape(B, self.num_heads, kv_seq_len, self.head_dim)

        # Scaled dot-product attention
        # Try Flash Attention (PyTorch 2.0+) for speed
        if self.use_flash_attn and hasattr(F, "scaled_dot_product_attention"):
            # Flash Attention: handles masking internally
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(attention_mask is None),
            )
        else:
            # Manual attention computation
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, kv_T]

            # Causal mask: prevent attending to future tokens
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.full((T, kv_seq_len), float("-inf"), device=x.device), diagonal=1 + offset
                )
                attn_scores = attn_scores + causal_mask
            else:
                attn_scores = attn_scores + attention_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, new_kv
