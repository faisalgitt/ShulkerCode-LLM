"""
=====================================
 Shulker Code — Main CLI Entrypoint
 Developed by @kopeedev / CyeroX
=====================================

Commands:
  python main.py generate  --prompt "..."
  python main.py train     --config configs/small.yaml --data data/
  python main.py debug     --file mycode.py
  python main.py webui
  python main.py info
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path

import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def cmd_generate(args):
    """Generate code from a prompt."""
    from utils.banner import print_banner, print_generation_header
    from utils.hardware import detect_hardware, get_device
    from model.transformer import ShulkerCodeModel, ShulkerConfig
    from data.tokenizer import ShulkerTokenizer
    from inference.engine import ShulkerInferenceEngine, GenerationConfig

    print_banner()

    # Load model
    device = args.device or "auto"
    model_path = args.model or "checkpoints/final"

    if not os.path.exists(model_path):
        console.print(f"[yellow]⚠️  No model found at '{model_path}'.[/]")
        console.print("[dim]   Using untrained nano model for demo...[/]")
        console.print("[dim]   Train first: python main.py train --config configs/nano.yaml[/]\n")

        # Create a fresh untrained nano model for demo
        config = ShulkerConfig(name="shulker-nano-demo")

        # Try to load a tokenizer, use simple fallback if not found
        try:
            tokenizer = ShulkerTokenizer.from_pretrained("tokenizer/")
        except Exception:
            tokenizer = _get_demo_tokenizer()

        model = ShulkerCodeModel(config)
    else:
        model = ShulkerCodeModel.from_pretrained(model_path, device=device)
        tokenizer = ShulkerTokenizer.from_pretrained(model_path)

    engine = ShulkerInferenceEngine(model, tokenizer, device=device)

    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    prompt = args.prompt
    lang = args.lang or "python"
    task = args.task or "gen"

    print_generation_header(prompt)

    console.print("[dim]Generating...[/]\n")

    # Stream output
    generated_tokens = []
    for token in engine.generate_streaming(prompt, config=config, language=lang, task=task):
        generated_tokens.append(token)
        console.print(token, end="", highlight=False)
        sys.stdout.flush()

    full_output = "".join(generated_tokens)
    console.print("\n")

    # Optionally save output
    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output)
        console.print(f"[green]✅ Output saved to {args.output}[/]")


def cmd_train(args):
    """Train a Shulker Code model."""
    from utils.banner import print_banner, print_hardware_info, print_training_start, print_model_info
    from utils.hardware import detect_hardware, setup_distributed
    from model.transformer import ShulkerCodeModel, ShulkerConfig
    from data.tokenizer import ShulkerTokenizer
    from data.dataset import CodeTokenDataset, StreamingCodeDataset, create_dataloader, discover_code_files
    from training.trainer import ShulkerTrainer

    print_banner()

    # Detect hardware
    hw = detect_hardware()
    print_hardware_info(hw)

    # Load config
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Build model config
    shulker_config = ShulkerConfig.from_dict(model_cfg)
    console.print(f"[cyan]🧱 Model: {shulker_config.name}[/]")

    # Build model
    model = ShulkerCodeModel(shulker_config)
    num_params = model.num_parameters()
    print_model_info(shulker_config, num_params)

    # Load or train tokenizer
    tokenizer_path = args.tokenizer or "tokenizer/"
    if os.path.exists(tokenizer_path):
        tokenizer = ShulkerTokenizer.from_pretrained(tokenizer_path)
    else:
        console.print("[yellow]⚠️  No tokenizer found. Using HuggingFace bootstrap tokenizer.[/]")
        tokenizer = ShulkerTokenizer.from_hf_pretrained("microsoft/codebert-base")
        tokenizer.save(tokenizer_path)

    # Discover dataset files
    data_path = args.data or "data/"
    if os.path.isdir(data_path):
        data_files = discover_code_files(data_path, extensions=[".jsonl", ".json", ".txt", ".py"])
    else:
        data_files = [data_path]

    if not data_files:
        console.print("[red]❌ No data files found! Add JSONL files to data/ directory.[/]")
        return

    # Create datasets
    max_seq = model_cfg.get("max_seq_len", 1024)
    dataset = CodeTokenDataset(
        file_paths=data_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=train_cfg.get("batch_size", 8),
        num_workers=train_cfg.get("dataloader_workers", 2),
    )

    # Setup distributed training
    rank, world_size, is_main = setup_distributed()

    # Train
    trainer = ShulkerTrainer(
        model=model,
        tokenizer=tokenizer,
        config=train_cfg,
        output_dir=args.output or "checkpoints",
        rank=rank,
        world_size=world_size,
    )

    print_training_start(train_cfg, train_cfg.get("max_steps", 10000))

    trainer.train(
        train_dataloader=dataloader,
        resume_from=args.resume,
    )


def cmd_debug(args):
    """Analyze a code file for bugs and suggest fixes."""
    from utils.banner import print_banner
    from utils.hardware import get_device
    from plugins.code_executor import SyntaxChecker, CodeExecutorPlugin

    print_banner()

    # Read code file
    if not os.path.exists(args.file):
        console.print(f"[red]❌ File not found: {args.file}[/]")
        return

    with open(args.file, "r") as f:
        code = f.read()

    # Determine language from extension
    ext_to_lang = {".py": "python", ".js": "javascript", ".ts": "typescript"}
    lang = ext_to_lang.get(Path(args.file).suffix.lower(), "python")

    console.print(Panel(
        Syntax(code, lang, theme="monokai", line_numbers=True),
        title=f"[bold]📄 {args.file}[/bold]",
        border_style="blue",
    ))

    # Syntax check (Python only)
    if lang == "python":
        checker = SyntaxChecker()
        valid, error = checker.check(code)
        if valid:
            console.print("[green]✅ Syntax: OK[/]")
        else:
            console.print(f"[red]❌ Syntax Error:\n{error}[/]")

    # Execute if requested
    if args.run and lang == "python":
        console.print("\n[cyan]▶️  Executing code...[/]")
        executor = CodeExecutorPlugin(timeout_seconds=10)
        success, stdout, stderr = executor.run_python(code)
        console.print(executor.format_result(success, stdout, stderr))

    # Generate fix suggestion with model (if available)
    model_path = args.model or "checkpoints/final"
    if os.path.exists(model_path):
        console.print("\n[cyan]🔍 Analyzing with Shulker Code...[/]")
        from model.transformer import ShulkerCodeModel
        from data.tokenizer import ShulkerTokenizer
        from inference.engine import ShulkerInferenceEngine, GenerationConfig

        model = ShulkerCodeModel.from_pretrained(model_path)
        tokenizer = ShulkerTokenizer.from_pretrained(model_path)
        engine = ShulkerInferenceEngine(model, tokenizer)

        prompt = f"# Fix the following {lang} code:\n{code}\n\n# Fixed version:\n"
        config = GenerationConfig(max_new_tokens=len(tokenizer.encode(code)) + 200, temperature=0.3)

        fixed = engine.generate(prompt, config=config, language=lang, task="fix")
        console.print(Panel(
            Syntax(fixed, lang, theme="monokai"),
            title="[bold green]🔧 Suggested Fix[/bold green]",
            border_style="green",
        ))
    else:
        console.print(f"\n[dim]💡 Train a model to get AI-powered fix suggestions.[/dim]")


def cmd_webui(args):
    """Launch the FastAPI web interface."""
    from utils.banner import print_banner
    print_banner()

    if args.model:
        os.environ["SHULKER_MODEL_PATH"] = args.model
    if args.device:
        os.environ["SHULKER_DEVICE"] = args.device

    console.print(f"[green]🌐 Starting Shulker Code Web UI...[/]")
    console.print(f"[dim]   Open: http://localhost:{args.port}[/]\n")

    import uvicorn
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def cmd_info(args):
    """Show system info and hardware profile."""
    from utils.banner import print_banner, print_hardware_info
    from utils.hardware import detect_hardware
    from utils.quantization import estimate_model_size

    print_banner()

    hw = detect_hardware()
    print_hardware_info(hw)

    # Show size estimates for all variants
    from rich.table import Table
    from rich import box

    table = Table(title="📐 Model Size Estimates", box=box.ROUNDED, border_style="cyan")
    table.add_column("Variant", style="bold cyan")
    table.add_column("Params")
    table.add_column("FP32 Size")
    table.add_column("FP16 Size")
    table.add_column("INT8 Size")
    table.add_column("Min VRAM")

    variants = [
        ("nano",   5_000_000),
        ("small",  125_000_000),
        ("medium", 1_300_000_000),
        ("large",  7_000_000_000),
        ("xlarge", 80_000_000_000),
    ]

    for name, params in variants:
        fp32 = estimate_model_size(params, "float32")
        fp16 = estimate_model_size(params, "float16")
        int8 = estimate_model_size(params, "int8")
        table.add_row(
            name,
            f"{params/1e6:.0f}M" if params < 1e9 else f"{params/1e9:.0f}B",
            f"{fp32:.1f} GB",
            f"{fp16:.1f} GB",
            f"{int8:.1f} GB",
            f"{fp16*1.2:.1f} GB",  # Rough estimate including activations
        )

    console.print(table)


def _get_demo_tokenizer():
    """Create a minimal demo tokenizer when none is installed."""
    class DemoTokenizer:
        vocab_size = 1000
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        def encode(self, text, **kwargs): return [ord(c) % 1000 for c in text[:100]]
        def encode_code(self, text, **kwargs): return self.encode(text)
        def decode(self, ids, **kwargs): return " ".join(str(i) for i in ids[:50])
    return DemoTokenizer()


def main():
    parser = argparse.ArgumentParser(
        prog="shulker",
        description="🧱 Shulker Code — Code Intelligence Engine by @kopeedev",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python main.py generate --prompt "Write a quicksort in Python"
          python main.py generate --prompt "Fix this code" --task fix --lang python
          python main.py train --config configs/nano.yaml --data data/
          python main.py train --config configs/small.yaml --data data/ --resume checkpoints/checkpoint-step_500
          python main.py debug --file mycode.py --run
          python main.py webui --port 8080
          python main.py info
        """),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────
    gen_p = subparsers.add_parser("generate", aliases=["gen", "g"], help="Generate code from a prompt")
    gen_p.add_argument("--prompt", "-p", required=True, help="Input prompt")
    gen_p.add_argument("--lang", default="python", help="Target language (default: python)")
    gen_p.add_argument("--task", default="gen", choices=["gen","fix","explain","optimize"])
    gen_p.add_argument("--model", default=None, help="Path to model checkpoint")
    gen_p.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    gen_p.add_argument("--max-tokens", type=int, default=512)
    gen_p.add_argument("--temperature", type=float, default=0.7)
    gen_p.add_argument("--top-k", type=int, default=50)
    gen_p.add_argument("--top-p", type=float, default=0.95)
    gen_p.add_argument("--output", "-o", default=None, help="Save output to file")
    gen_p.set_defaults(func=cmd_generate)

    # ── train ─────────────────────────────────
    train_p = subparsers.add_parser("train", aliases=["t"], help="Train a Shulker Code model")
    train_p.add_argument("--config", "-c", required=True, help="Path to YAML config (e.g. configs/nano.yaml)")
    train_p.add_argument("--data", "-d", default="data/", help="Data directory or file path")
    train_p.add_argument("--tokenizer", default=None, help="Path to tokenizer (optional)")
    train_p.add_argument("--output", default="checkpoints/", help="Checkpoint output directory")
    train_p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    train_p.set_defaults(func=cmd_train)

    # ── debug ─────────────────────────────────
    debug_p = subparsers.add_parser("debug", aliases=["d"], help="Debug and analyze a code file")
    debug_p.add_argument("--file", "-f", required=True, help="Source code file to analyze")
    debug_p.add_argument("--run", action="store_true", help="Execute the code (Python only)")
    debug_p.add_argument("--model", default=None)
    debug_p.add_argument("--device", default="auto")
    debug_p.set_defaults(func=cmd_debug)

    # ── webui ─────────────────────────────────
    web_p = subparsers.add_parser("webui", aliases=["web", "w"], help="Launch the Web UI")
    web_p.add_argument("--host", default="0.0.0.0")
    web_p.add_argument("--port", type=int, default=8000)
    web_p.add_argument("--model", default=None)
    web_p.add_argument("--device", default="auto")
    web_p.set_defaults(func=cmd_webui)

    # ── info ──────────────────────────────────
    info_p = subparsers.add_parser("info", help="Show hardware info and model size estimates")
    info_p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
