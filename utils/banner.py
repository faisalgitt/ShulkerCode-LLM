"""
=====================================
 Shulker Code — CLI Banner
 Developed by @kopeedev / CyeroX
=====================================
"""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ASCII Art Banner for Shulker Code
SHULKER_BANNER = r"""
 _____ _           _ _             ____          _
/ ____| |         | | |           / ___|___   __| | ___
\\___ \| |__  _   _| | | _____ _ _| |   / _ \\ / _` |/ _ \\
 ___) | '_ \\| | | | | |/ / _ \\ '__| |__| (_) | (_| |  __/
|____/|_| |_|\\__,_|_|_|_<  __/ |  \\____\\___/ \\__,_|\\___|
                          \\___|_|
"""


def print_banner(version: str = "1.0.0"):
    """Print the Shulker Code CLI banner."""
    console.print()

    # Gradient-like colored banner
    banner_lines = SHULKER_BANNER.strip().split("\n")
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_magenta", "magenta"]

    for i, line in enumerate(banner_lines):
        color = colors[i % len(colors)]
        console.print(f"  [bold {color}]{line}[/]")

    console.print()
    console.print(f"  [dim]v{version}  ·  Code Intelligence Engine  ·  by [bold cyan]@kopeedev[/bold cyan] / CyeroX Dev[/dim]")
    console.print()


def print_hardware_info(profile):
    """Print a hardware detection summary."""
    table = Table(
        title="🖥️  Hardware Profile",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
    )
    table.add_column("Property", style="dim", width=18)
    table.add_column("Value", style="bold")

    table.add_row("Device", f"{profile.device_type.upper()} × {profile.device_count}")
    table.add_row("VRAM", f"{profile.total_vram_gb:.1f} GB")
    table.add_row("RAM", f"{profile.total_ram_gb:.1f} GB")
    table.add_row("FP16", "✅ Yes" if profile.supports_fp16 else "❌ No")
    table.add_row("BF16", "✅ Yes" if profile.supports_bf16 else "❌ No")
    table.add_row("Flash Attn", "✅ Yes" if profile.supports_flash_attn else "❌ No")
    table.add_row("Recommended", f"[bold green]shulker-{profile.recommended_variant}[/]")
    table.add_row("Quantization", f"[yellow]{profile.recommended_quant or 'None'}[/]")

    console.print(table)
    console.print()


def print_generation_header(prompt: str, model_name: str = "shulker-nano"):
    """Print header before generation output."""
    console.print(Panel(
        f"[dim]{prompt[:100]}{'...' if len(prompt) > 100 else ''}[/dim]",
        title=f"[bold cyan]🧱 {model_name}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


def print_model_info(config, num_params: int):
    """Print model architecture summary."""
    table = Table(
        title=f"🧠 {config.name} Architecture",
        box=box.ROUNDED,
        border_style="magenta",
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Total Parameters", f"{num_params:,}")
    table.add_row("Layers", str(config.num_layers))
    table.add_row("Attention Heads", str(config.num_heads))
    table.add_row("KV Heads", str(config.num_kv_heads))
    table.add_row("Hidden Size", str(config.hidden_size))
    table.add_row("FFN Size", str(config.intermediate_size))
    table.add_row("Max Seq Length", str(config.max_seq_len))
    table.add_row("Vocab Size", f"{config.vocab_size:,}")
    table.add_row("RoPE Theta", str(config.rope_theta))

    console.print(table)
    console.print()


def print_training_start(config: dict, total_steps: int):
    """Print training start summary."""
    console.print(Panel(
        f"[bold green]🚀 Training started![/]\n"
        f"  Batch size:   [cyan]{config.get('batch_size', '?')}[/]\n"
        f"  Max steps:    [cyan]{total_steps:,}[/]\n"
        f"  Learning rate:[cyan]{config.get('learning_rate', '?')}[/]\n"
        f"  Mixed prec:   [cyan]{'BF16' if config.get('bf16') else 'FP16' if config.get('fp16') else 'FP32'}[/]",
        title="[bold]Training Configuration[/]",
        border_style="green",
    ))
    console.print()


def print_success(message: str):
    console.print(f"[bold green]✅ {message}[/]")


def print_error(message: str):
    console.print(f"[bold red]❌ {message}[/]")


def print_warning(message: str):
    console.print(f"[bold yellow]⚠️  {message}[/]")


def print_info(message: str):
    console.print(f"[cyan]ℹ️  {message}[/]")
