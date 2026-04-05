"""
=====================================
 Shulker Code — Hardware Utilities
 Auto-detect and optimize for device
 Developed by @kopeedev / CyeroX
=====================================
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HardwareProfile:
    """Detected hardware profile for optimizing model loading."""
    device_type: str          # "cpu", "cuda", "mps"
    device_count: int         # Number of GPUs
    total_vram_gb: float      # Total VRAM in GB
    total_ram_gb: float       # Total system RAM in GB
    supports_fp16: bool       # Hardware FP16 support
    supports_bf16: bool       # Hardware BF16 support
    supports_flash_attn: bool # Flash Attention support
    recommended_variant: str  # "nano", "small", "medium", "large", "xlarge"
    recommended_quant: Optional[str]  # "int8", "int4", or None

    def summary(self) -> str:
        lines = [
            f"  Device:       {self.device_type.upper()} × {self.device_count}",
            f"  VRAM:         {self.total_vram_gb:.1f} GB",
            f"  RAM:          {self.total_ram_gb:.1f} GB",
            f"  FP16:         {'✅' if self.supports_fp16 else '❌'}",
            f"  BF16:         {'✅' if self.supports_bf16 else '❌'}",
            f"  Flash Attn:   {'✅' if self.supports_flash_attn else '❌'}",
            f"  Recommended:  shulker-{self.recommended_variant}",
            f"  Quantization: {self.recommended_quant or 'None (full precision)'}",
        ]
        return "\n".join(lines)


def detect_hardware() -> HardwareProfile:
    """
    Auto-detect hardware and return an optimized profile.
    Detects: CPU, CUDA GPU(s), Apple Silicon MPS.
    """
    import psutil

    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    total_vram_gb = 0.0
    device_type = "cpu"
    device_count = 0
    supports_fp16 = False
    supports_bf16 = False
    supports_flash_attn = False

    # Check CUDA
    if torch.cuda.is_available():
        device_type = "cuda"
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb += props.total_memory / (1024 ** 3)
            # BF16 requires Ampere (sm_80) or newer
            if props.major >= 8:
                supports_bf16 = True
            # FP16 requires Volta (sm_70) or newer
            if props.major >= 7:
                supports_fp16 = True

        # Flash Attention available in PyTorch 2.0+
        supports_flash_attn = (
            int(torch.__version__.split(".")[0]) >= 2
        )

    # Check Apple Silicon MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
        device_count = 1
        # MPS doesn't expose VRAM separately; approximate from RAM
        total_vram_gb = total_ram_gb * 0.7  # Unified memory
        supports_fp16 = True
        supports_bf16 = False

    # Determine recommended variant based on VRAM/RAM
    if device_type == "cuda":
        if total_vram_gb >= 80:
            recommended_variant = "xlarge"
            recommended_quant = None
        elif total_vram_gb >= 40:
            recommended_variant = "large"
            recommended_quant = None
        elif total_vram_gb >= 20:
            recommended_variant = "medium"
            recommended_quant = None
        elif total_vram_gb >= 8:
            recommended_variant = "small"
            recommended_quant = None
        elif total_vram_gb >= 4:
            recommended_variant = "small"
            recommended_quant = "int8"
        else:
            recommended_variant = "nano"
            recommended_quant = "int8"
    elif device_type == "mps":
        if total_ram_gb >= 32:
            recommended_variant = "small"
            recommended_quant = None
        else:
            recommended_variant = "nano"
            recommended_quant = None
    else:
        # CPU only
        if total_ram_gb >= 32:
            recommended_variant = "nano"
            recommended_quant = "int8"
        else:
            recommended_variant = "nano"
            recommended_quant = "int8"

    return HardwareProfile(
        device_type=device_type,
        device_count=device_count,
        total_vram_gb=total_vram_gb,
        total_ram_gb=total_ram_gb,
        supports_fp16=supports_fp16,
        supports_bf16=supports_bf16,
        supports_flash_attn=supports_flash_attn,
        recommended_variant=recommended_variant,
        recommended_quant=recommended_quant,
    )


def get_device(device: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.
    "auto" → picks best available device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def setup_distributed() -> Tuple[int, int, bool]:
    """
    Initialize distributed training (DDP).
    Returns: (rank, world_size, is_main_process)
    """
    if "RANK" not in os.environ:
        return 0, 1, True

    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size, rank == 0


def get_dtype(fp16: bool = False, bf16: bool = False) -> torch.dtype:
    """Get the appropriate tensor dtype."""
    if bf16:
        return torch.bfloat16
    elif fp16:
        return torch.float16
    return torch.float32


def model_to_device(model: nn.Module, device: str = "auto", dtype: Optional[torch.dtype] = None) -> nn.Module:
    """Move model to target device with optional dtype conversion."""
    target_device = get_device(device)
    model = model.to(target_device)
    if dtype is not None:
        model = model.to(dtype)
    return model


def memory_stats() -> Dict[str, float]:
    """Return current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def clear_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
