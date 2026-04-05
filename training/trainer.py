"""
=====================================
 Shulker Code — Trainer
 Full training loop with all features
 Developed by @kopeedev / CyeroX
=====================================

Features:
 - Gradient accumulation (train with small VRAM)
 - Mixed precision (FP16/BF16)
 - Checkpoint saving & loading (resume training)
 - Cosine LR schedule with warmup
 - Multi-GPU support (DDP / FSDP)
 - Training metrics logging
"""

import os
import time
import json
import math
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


class CosineSchedulerWithWarmup:
    """
    Cosine LR scheduler with linear warmup.
    This is the standard schedule used in modern LLM training.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_scale = self._get_lr_scale(self.current_step)
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * lr_scale

    def _get_lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return step / max(1, self.warmup_steps)
        elif step >= self.max_steps:
            return self.min_lr_ratio
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict):
        self.current_step = state["current_step"]
        # Restore LR for current step
        lr_scale = self._get_lr_scale(self.current_step)
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * lr_scale


class ShulkerTrainer:
    """
    Production-grade training loop for Shulker Code.

    Usage:
        trainer = ShulkerTrainer(model, tokenizer, config)
        trainer.train(train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Dict[str, Any],
        output_dir: str = "checkpoints",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        os.makedirs(output_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # Mixed precision
        self.use_amp = config.get("fp16", False) or config.get("bf16", False)
        self.amp_dtype = torch.bfloat16 if config.get("bf16") else torch.float16
        self.scaler = GradScaler() if config.get("fp16") and torch.cuda.is_available() else None

        # Wrap for multi-GPU
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Metrics log
        self.metrics_log: list = []

    def _build_optimizer(self) -> AdamW:
        """
        Build AdamW optimizer with weight decay only on non-bias/norm params.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and norms
            if name.endswith(".bias") or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config.get("learning_rate", 3e-4),
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _build_scheduler(self, total_steps: int) -> CosineSchedulerWithWarmup:
        """Build the learning rate scheduler."""
        return CosineSchedulerWithWarmup(
            optimizer=self.optimizer,
            warmup_steps=self.config.get("warmup_steps", 100),
            max_steps=total_steps,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data
            val_dataloader: Optional validation data
            resume_from: Path to checkpoint to resume from
        """
        max_steps = self.config.get("max_steps", 10000)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        save_steps = self.config.get("save_steps", 500)
        eval_steps = self.config.get("eval_steps", 250)
        log_steps = self.config.get("log_steps", 10)

        scheduler = self._build_scheduler(max_steps)

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from, scheduler)
            if self.is_main:
                print(f"▶️  Resuming training from step {self.global_step}")

        if self.is_main:
            print(f"🏋️  Starting training for {max_steps:,} steps")
            print(f"   Gradient accumulation: {grad_accum} steps")
            print(f"   Effective batch size:  {self.config.get('batch_size', 1) * grad_accum * self.world_size}")

        self.model.train()
        self.optimizer.zero_grad()

        # Training loop
        train_iter = iter(train_dataloader)
        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.is_main,
            dynamic_ncols=True,
        )

        accum_loss = 0.0
        accum_steps = 0

        while self.global_step < max_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart dataloader for new epoch
                self.epoch += 1
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with optional mixed precision
            if self.use_amp and torch.cuda.is_available():
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] / grad_accum
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / grad_accum

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            accum_steps += 1

            # Update weights after accumulating enough gradients
            if accum_steps % grad_accum == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Log metrics
                if self.global_step % log_steps == 0 and self.is_main:
                    lr = scheduler.get_last_lr()[0]
                    actual_loss = accum_loss * grad_accum  # Undo the accum division
                    pbar.set_postfix({
                        "loss": f"{actual_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{grad_norm:.2f}",
                    })
                    self.metrics_log.append({
                        "step": self.global_step,
                        "loss": actual_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                    })
                    accum_loss = 0.0

                pbar.update(1)

                # Evaluation
                if val_dataloader and self.global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_dataloader)
                    if self.is_main:
                        pbar.write(f"📊 Step {self.global_step} | Val Loss: {val_loss:.4f}")
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint("best", scheduler)
                            pbar.write(f"   🏆 New best model saved!")
                    self.model.train()

                # Save checkpoint
                if self.global_step % save_steps == 0 and self.is_main:
                    self._save_checkpoint(f"step_{self.global_step}", scheduler)
                    pbar.write(f"💾 Checkpoint saved at step {self.global_step}")

        pbar.close()

        # Save final model
        if self.is_main:
            self._save_checkpoint("final", scheduler)
            self._save_metrics()
            print(f"\n✅ Training complete! {self.global_step:,} steps")

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> float:
        """Evaluate on validation set. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False, disable=not self.is_main):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.use_amp and torch.cuda.is_available():
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                    )
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            total_batches += 1

        return total_loss / max(1, total_batches)

    def _save_checkpoint(self, tag: str, scheduler: CosineSchedulerWithWarmup):
        """Save a training checkpoint."""
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Get underlying model (unwrap DDP if needed)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(ckpt_dir)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
        }, os.path.join(ckpt_dir, "training_state.pt"))

    def _load_checkpoint(self, checkpoint_path: str, scheduler: CosineSchedulerWithWarmup):
        """Load a training checkpoint to resume training."""
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.exists(state_path):
            print(f"⚠️  No training state found at {state_path}")
            return

        state = torch.load(state_path, map_location=self.device)
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if self.scaler and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

        # Load model weights
        model = self.model.module if hasattr(self.model, "module") else self.model
        weights_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(weights)

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        print(f"📊 Metrics saved to {metrics_path}")
