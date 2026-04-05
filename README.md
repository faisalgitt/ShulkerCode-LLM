# 🧱 Shulker Code — Coding LLM by @kopeedev

> A production-ready, scalable Transformer-based LLM built **exclusively for programming tasks**.  
> From 5M to 80B parameters. CPU to multi-GPU. Your local GitHub Copilot.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate code
python main.py generate --prompt "Write a Python function to reverse a linked list"

# Train on your own dataset
python main.py train --config configs/small.yaml --data data/sample_dataset.jsonl

# Debug mode (analyze + fix code)
python main.py debug --file mycode.py

# Launch Web UI
python main.py webui
```

---

## 📐 Model Variants

| Variant    | Params   | Layers | Heads | Hidden | Use Case             |
|------------|----------|--------|-------|--------|----------------------|
| `nano`     | ~5M      | 4      | 4     | 256    | CPU / Edge devices   |
| `small`    | ~125M    | 12     | 12    | 768    | Laptop GPU           |
| `medium`   | ~1.3B    | 24     | 16    | 2048   | RTX 3080+            |
| `large`    | ~7B      | 32     | 32    | 4096   | A100 / Multi-GPU     |
| `xlarge`   | ~80B     | 64     | 64    | 8192   | GPU Clusters         |

---

## 📁 Project Structure

```
shulker_code/
├── model/              # Transformer architecture
│   ├── transformer.py  # Core model (GPT-style decoder)
│   ├── attention.py    # Multi-head & grouped-query attention
│   ├── embeddings.py   # Token + positional embeddings (RoPE)
│   └── lora.py         # LoRA / PEFT fine-tuning
├── data/               # Data pipeline
│   ├── tokenizer.py    # BPE tokenizer (code-optimized)
│   ├── dataset.py      # Dataset loaders (JSON/TXT/JSONL)
│   └── sample_dataset.jsonl
├── training/           # Training system
│   ├── trainer.py      # Main training loop
│   ├── scheduler.py    # LR schedulers
│   └── checkpointing.py
├── inference/          # Inference engine
│   ├── engine.py       # Fast generation engine
│   └── streaming.py    # Streaming / typing-effect output
├── utils/              # Utilities
│   ├── hardware.py     # Hardware detection & optimization
│   ├── quantization.py # INT8 / 4-bit quantization
│   └── banner.py       # CLI ASCII art banner
├── configs/            # Model configs
│   ├── nano.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   └── xlarge.yaml
├── plugins/            # Plugin system
│   └── code_executor.py
├── web/                # Web UI
│   └── app.py          # FastAPI web interface
├── main.py             # CLI entrypoint
└── requirements.txt
```

---

## 🛠 Hardware Guide

- **CPU only** → use `nano` config with INT8 quantization  
- **Single GPU (8GB)** → use `small` with FP16  
- **Single GPU (24GB+)** → use `medium` with FP16  
- **Multi-GPU** → use `large` or `xlarge` with DDP/FSDP  

---

## 📜 License
MIT — Developed by [@kopeedev](https://github.com/faisalgitt) / CyeroX Development
