"""
=====================================
 Shulker Code — LoRA / PEFT Module
 Low-Rank Adaptation for fine-tuning
 Developed by @kopeedev / CyeroX
=====================================

LoRA: Instead of updating all weights W, we learn two small matrices:
    W' = W + α * (A @ B)
where A ∈ R^{d×r} and B ∈ R^{r×k}, rank r << d,k

This makes fine-tuning with minimal parameters possible!
"""

import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A Linear layer augmented with LoRA adapters.
    Wraps an existing nn.Linear and adds low-rank update matrices.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        enabled: bool = True,
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = enabled

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Keep the original frozen weights
        self.weight = original_layer.weight
        self.bias = original_layer.bias

        if enabled and rank > 0:
            # LoRA matrices: A (init with Kaiming), B (init with zeros)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_dropout = nn.Dropout(dropout)

            # Initialize A with kaiming uniform (standard)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen forward pass
        base_out = F.linear(x, self.weight, self.bias)

        # Add LoRA delta (if enabled)
        if self.enabled and self.rank > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_out + lora_out * self.scaling

        return base_out

    def merge_weights(self):
        """
        Merge LoRA weights into the base weights permanently.
        Call before exporting/deploying to remove LoRA overhead.
        """
        if self.enabled and self.rank > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta
            self.lora_A = None
            self.lora_B = None
            self.enabled = False
            print("✅ LoRA weights merged into base layer.")


class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        # Which modules to apply LoRA to
        target_modules: Optional[List[str]] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        # Default: apply to attention projection layers
        self.target_modules = target_modules or ["q_proj", "v_proj", "o_proj"]


def apply_lora(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA adapters to a model's target linear layers.

    Steps:
    1. Freeze ALL base model parameters
    2. Replace target Linear layers with LoRALinear wrappers
    3. Only LoRA parameters (A, B) are trainable

    Args:
        model: The base ShulkerCodeModel
        lora_config: LoRA configuration

    Returns:
        Modified model with LoRA applied
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Replace target layers
    replaced = 0
    for name, module in model.named_modules():
        for target in lora_config.target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent module
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]

                # Replace with LoRA wrapper
                lora_layer = LoRALinear(
                    original_layer=module,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                )
                setattr(parent, child_name, lora_layer)
                replaced += 1
                break

    # Step 3: Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA applied to {replaced} layers")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA adapter weights (very small files!)."""
    import os
    os.makedirs(path, exist_ok=True)
    lora_state = {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    torch.save(lora_state, os.path.join(path, "lora_weights.pt"))
    print(f"✅ LoRA weights saved to {path} ({len(lora_state)} tensors)")


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """Load LoRA adapter weights back into a model."""
    import os
    weights_path = os.path.join(path, "lora_weights.pt")
    lora_state = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"✅ LoRA weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return model
