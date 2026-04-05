"""
=====================================
 Shulker Code — Dataset Module
 Code dataset loaders for training
 Developed by @kopeedev / CyeroX
=====================================

Supported formats:
 - JSONL: {"text": "...", "lang": "python"} or {"prompt": "...", "completion": "..."}
 - TXT: Plain code files
 - HuggingFace datasets (GitHub Code, CodeSearchNet, etc.)
"""

import os
import json
import glob
import random
from typing import Optional, List, Dict, Iterator, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ─────────────────────────────────────────────
# Core Token Dataset
# ─────────────────────────────────────────────
class CodeTokenDataset(Dataset):
    """
    Dataset that loads pre-tokenized code sequences from JSONL files.

    Each sample in the JSONL can have:
    - {"text": "..."} → raw code/text to tokenize
    - {"input_ids": [1, 2, ...]} → pre-tokenized (faster)
    - {"prompt": "...", "completion": "..."} → instruction format

    Uses sliding window to handle long files.
    """

    def __init__(
        self,
        file_paths: Union[str, List[str]],
        tokenizer,
        max_seq_len: int = 1024,
        stride: int = 512,        # Sliding window stride
        task: str = "gen",
        language: str = "python",
        cache_tokenized: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.task = task
        self.language = language

        # Normalize to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Load all samples
        self.samples: List[List[int]] = []
        self._load_files(file_paths, cache_tokenized)

        print(f"📦 Dataset loaded: {len(self.samples):,} sequences from {len(file_paths)} files")

    def _load_files(self, file_paths: List[str], cache_tokenized: bool):
        """Load and tokenize all files."""
        for path in file_paths:
            if not os.path.exists(path):
                print(f"⚠️  File not found: {path}")
                continue

            ext = Path(path).suffix.lower()

            if ext in (".jsonl", ".json"):
                self._load_jsonl(path)
            elif ext in (".txt", ".py", ".js", ".ts", ".cpp", ".java", ".go", ".rs"):
                self._load_text_file(path)
            else:
                # Try as text
                self._load_text_file(path)

    def _load_jsonl(self, path: str):
        """Load a JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self._add_item(item)
                except json.JSONDecodeError:
                    continue

    def _load_text_file(self, path: str):
        """Load a plain text/code file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Detect language from extension
        ext_to_lang = {".py": "python", ".js": "javascript", ".ts": "typescript",
                       ".cpp": "cpp", ".java": "java", ".go": "go", ".rs": "rust"}
        lang = ext_to_lang.get(Path(path).suffix.lower(), "text")

        self._add_text(text, lang)

    def _add_item(self, item: Dict):
        """Process a single dataset item."""
        if "input_ids" in item:
            # Already tokenized
            ids = item["input_ids"]
            self._add_ids(ids)
        elif "text" in item:
            lang = item.get("lang", self.language)
            self._add_text(item["text"], lang)
        elif "prompt" in item and "completion" in item:
            # Instruction-following format
            text = item["prompt"] + item["completion"]
            lang = item.get("lang", self.language)
            self._add_text(text, lang)
        elif "content" in item:
            # GitHub Code format
            lang = item.get("programming_language", self.language)
            self._add_text(item["content"], lang)

    def _add_text(self, text: str, lang: str = "python"):
        """Tokenize text and add as sliding window chunks."""
        try:
            ids = self.tokenizer.encode_code(text, language=lang, task=self.task)
            self._add_ids(ids)
        except Exception:
            # Fallback: basic encode
            try:
                ids = self.tokenizer.encode(text)
                self._add_ids(ids)
            except Exception:
                pass

    def _add_ids(self, ids: List[int]):
        """Split token sequence into overlapping windows."""
        if len(ids) < 2:
            return

        # Slide a window over the sequence
        for start in range(0, max(1, len(ids) - self.max_seq_len + 1), self.stride):
            chunk = ids[start : start + self.max_seq_len]
            if len(chunk) >= 2:  # Need at least 2 tokens for input/label pairs
                self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[idx]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),  # Same as input for CLM
        }


# ─────────────────────────────────────────────
# Streaming Dataset (for huge datasets)
# ─────────────────────────────────────────────
class StreamingCodeDataset(IterableDataset):
    """
    Streaming dataset for training on massive code corpora.
    Reads from disk on-the-fly without loading everything into RAM.
    Perfect for datasets like GitHub Code (hundreds of GB).
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        max_seq_len: int = 1024,
        shuffle: bool = True,
        buffer_size: int = 1000,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(files)

        buffer = []

        for path in files:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        text = item.get("text") or item.get("content") or item.get("code", "")
                        lang = item.get("lang", "python")
                    except Exception:
                        text = line
                        lang = "python"

                    if not text:
                        continue

                    try:
                        ids = self.tokenizer.encode_code(text, language=lang)
                    except Exception:
                        continue

                    # Yield chunks
                    for start in range(0, max(1, len(ids) - self.max_seq_len + 1), self.max_seq_len // 2):
                        chunk = ids[start : start + self.max_seq_len]
                        if len(chunk) < 2:
                            continue

                        pad_len = self.max_seq_len - len(chunk)
                        attention_mask = [1] * len(chunk) + [0] * pad_len
                        chunk = chunk + [self.tokenizer.pad_token_id] * pad_len

                        sample = {
                            "input_ids": torch.tensor(chunk, dtype=torch.long),
                            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                            "labels": torch.tensor(chunk, dtype=torch.long),
                        }

                        buffer.append(sample)

                        if self.shuffle and len(buffer) >= self.buffer_size:
                            random.shuffle(buffer)
                            yield from buffer
                            buffer = []

        # Yield remaining
        if self.shuffle:
            random.shuffle(buffer)
        yield from buffer


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create an optimized DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not isinstance(dataset, IterableDataset),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def discover_code_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Recursively find all code files in a directory.

    Args:
        directory: Root directory to search
        extensions: List of extensions, e.g. [".py", ".js"]

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".cpp", ".java", ".go", ".rs", ".cs", ".txt", ".jsonl"]

    files = []
    for ext in extensions:
        pattern = os.path.join(directory, "**", f"*{ext}")
        files.extend(glob.glob(pattern, recursive=True))

    print(f"🔍 Found {len(files):,} code files in {directory}")
    return files
