"""
=====================================
 Shulker Code — Quantization Module
 INT8 / INT4 model quantization
 Developed by @kopeedev / CyeroX
=====================================

Quantization reduces model size and speeds up CPU inference:
- INT8: ~2x smaller, ~1.5x faster, minimal quality loss
- INT4: ~4x smaller, ~2x faster, some quality loss

On CPU devices with limited RAM, always use INT8 quantization.
"""

import torch
import torch.nn as nn
from typing import Optional


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to a model.
    Works on CPU only. Replaces Linear layers with INT8 versions.

    Args:
        model: The model to quantize

    Returns:
        Quantized model (in-place + returned)
    """
    print("⚙️  Applying INT8 dynamic quantization...")
    model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},  # Quantize only Linear layers
        dtype=torch.qint8,
    )
    print("✅ INT8 quantization applied")
    return model


def quantize_model_int4(model: nn.Module, device: str = "auto") -> nn.Module:
    """
    Apply 4-bit quantization using bitsandbytes library.
    Supports both CPU and CUDA.

    Requires: pip install bitsandbytes

    Args:
        model: The model to quantize
        device: Target device

    Returns:
        4-bit quantized model
    """
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("⚠️  bitsandbytes not found. Falling back to INT8 quantization.")
        return quantize_model_int8(model)

    print("⚙️  Applying INT4 quantization (bitsandbytes)...")

    # Replace Linear layers with 4-bit versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 4096:
            # Replace with 4-bit linear
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = dict(model.named_modules()).get(parent_name, model)

            try:
                quant_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=True,
                    quant_type="nf4",  # NF4: better quality than int4
                )
                quant_layer.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                )
                setattr(parent, child_name, quant_layer)
            except Exception as e:
                # Skip layers that can't be quantized
                pass

    print("✅ INT4 (NF4) quantization applied")
    return model


def estimate_model_size(num_params: int, dtype: str = "float32") -> float:
    """
    Estimate model size in GB.

    Args:
        num_params: Number of parameters
        dtype: "float32" (4B), "float16"/"bfloat16" (2B), "int8" (1B), "int4" (0.5B)

    Returns:
        Size in GB
    """
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }.get(dtype, 4)

    size_bytes = num_params * bytes_per_param
    return size_bytes / (1024 ** 3)


def print_model_size(model: nn.Module, name: str = "Model"):
    """Print model parameter count and size estimates."""
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 {name} Size Report:")
    print(f"   Total params:     {num_params:>15,}")
    print(f"   Trainable params: {trainable:>15,}")
    print(f"   FP32 size:        {estimate_model_size(num_params, 'float32'):>10.2f} GB")
    print(f"   FP16 size:        {estimate_model_size(num_params, 'float16'):>10.2f} GB")
    print(f"   INT8 size:        {estimate_model_size(num_params, 'int8'):>10.2f} GB")
    print(f"   INT4 size:        {estimate_model_size(num_params, 'int4'):>10.2f} GB")
