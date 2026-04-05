"""
=====================================
 Shulker Code — Tokenizer
 BPE tokenizer optimized for code
 Developed by @kopeedev / CyeroX
=====================================

This tokenizer is specifically designed for programming languages:
- Preserves indentation (important for Python!)
- Handles special tokens: <code>, </code>, <lang:python>, etc.
- Supports 50+ programming languages
- BPE (Byte Pair Encoding) vocabulary
"""

import os
import json
import regex as re
from typing import List, Optional, Dict, Union
from pathlib import Path


# ─────────────────────────────────────────────
# Special tokens used by Shulker Code
# ─────────────────────────────────────────────
SPECIAL_TOKENS = {
    "<pad>":      0,
    "<bos>":      1,
    "<eos>":      2,
    "<unk>":      3,
    "<sep>":      4,
    "<mask>":     5,
    # Code-specific tokens
    "<code>":     6,
    "</code>":    7,
    "<comment>":  8,
    "</comment>": 9,
    # Language markers
    "<lang:py>":  10,
    "<lang:js>":  11,
    "<lang:cpp>": 12,
    "<lang:java>":13,
    "<lang:ts>":  14,
    "<lang:go>":  15,
    "<lang:rs>":  16,
    "<lang:cs>":  17,
    # Task markers
    "<task:gen>": 18,
    "<task:fix>": 19,
    "<task:explain>": 20,
    "<task:optimize>": 21,
}

# Language extension to token mapping
LANG_TOKENS = {
    ".py":   "<lang:py>",
    ".js":   "<lang:js>",
    ".ts":   "<lang:ts>",
    ".cpp":  "<lang:cpp>",
    ".cc":   "<lang:cpp>",
    ".java": "<lang:java>",
    ".go":   "<lang:go>",
    ".rs":   "<lang:rs>",
    ".cs":   "<lang:cs>",
    ".rb":   "<lang:rb>",
    ".php":  "<lang:php>",
}


class ShulkerTokenizer:
    """
    BPE tokenizer for Shulker Code.

    Uses the HuggingFace `tokenizers` library under the hood,
    but wraps it with code-specific pre/post processing.

    For training a tokenizer from scratch:
        tokenizer = ShulkerTokenizer.train_from_files(files, vocab_size=32000)

    For loading a pretrained tokenizer:
        tokenizer = ShulkerTokenizer.from_pretrained("path/to/tokenizer")
    """

    def __init__(self):
        self._tokenizer = None
        self.special_tokens = SPECIAL_TOKENS
        self.vocab_size = 32000

        # Code-aware pre-tokenization regex
        # Splits on whitespace, but preserves indentation
        self._code_pattern = re.compile(
            r"""(?x)
            \s+                    # whitespace (preserved for indentation)
            | [a-zA-Z_]\w*         # identifiers
            | \d+(?:\.\d+)?        # numbers
            | "(?:[^"\\]|\\.)*"    # double-quoted strings
            | '(?:[^'\\]|\\.)*'    # single-quoted strings
            | `(?:[^`\\]|\\.)*`    # backtick strings
            | (?:\/\/|#)[^\n]*     # comments
            | [+\-*/=<>!&|^~%@]+   # operators
            | [{}()\[\];:,.]       # delimiters
            | .                    # fallback
            """
        )

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        save_path: Optional[str] = None,
    ) -> "ShulkerTokenizer":
        """
        Train a new BPE tokenizer from code files.

        Args:
            files: List of .py, .js, .txt, .jsonl file paths
            vocab_size: Target vocabulary size
            min_frequency: Minimum token frequency
            save_path: Optional directory to save tokenizer

        Returns:
            Trained ShulkerTokenizer instance
        """
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
        except ImportError:
            raise ImportError("Install tokenizers: pip install tokenizers")

        instance = cls()

        # Build BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Code-aware pre-tokenizer: byte-level BPE
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # BPE trainer configuration
        special_token_list = list(SPECIAL_TOKENS.keys())
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_token_list,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        # Train on files
        print(f"🔤 Training BPE tokenizer on {len(files)} files...")
        tokenizer.train(files, trainer)

        # Add byte-level decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor for BOS/EOS
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        instance._tokenizer = tokenizer
        instance.vocab_size = tokenizer.get_vocab_size()

        if save_path:
            instance.save(save_path)
            print(f"✅ Tokenizer saved to {save_path}")

        print(f"✅ Tokenizer trained: {instance.vocab_size:,} tokens")
        return instance

    @classmethod
    def from_pretrained(cls, path: str) -> "ShulkerTokenizer":
        """Load a saved tokenizer from disk."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("Install tokenizers: pip install tokenizers")

        instance = cls()
        tokenizer_path = os.path.join(path, "tokenizer.json")
        instance._tokenizer = Tokenizer.from_file(tokenizer_path)
        instance.vocab_size = instance._tokenizer.get_vocab_size()

        # Load vocab size override if present
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                instance.vocab_size = config.get("vocab_size", instance.vocab_size)

        print(f"✅ Tokenizer loaded from {path} ({instance.vocab_size:,} tokens)")
        return instance

    @classmethod
    def from_hf_pretrained(cls, model_name: str = "microsoft/codebert-base") -> "ShulkerTokenizer":
        """
        Bootstrap from a HuggingFace pretrained tokenizer.
        Useful for quick setup without training from scratch.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        instance = cls()
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Wrap HF tokenizer with our interface
        instance._hf_tokenizer = hf_tokenizer
        instance._use_hf = True
        instance.vocab_size = hf_tokenizer.vocab_size

        print(f"✅ Using HuggingFace tokenizer: {model_name} ({instance.vocab_size:,} tokens)")
        return instance

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        lang: Optional[str] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text/code
            add_special_tokens: Wrap with <bos>/<eos>
            max_length: Truncate to this length
            lang: Optional language hint (e.g. "python", "javascript")

        Returns:
            List of integer token IDs
        """
        if hasattr(self, "_use_hf") and self._use_hf:
            result = self._hf_tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=max_length is not None,
            )
            return result

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call train_from_files() or from_pretrained() first.")

        encoding = self._tokenizer.encode(text)
        ids = encoding.ids

        if add_special_tokens:
            ids = [self.special_tokens["<bos>"]] + ids + [self.special_tokens["<eos>"]]

        if max_length:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if hasattr(self, "_use_hf") and self._use_hf:
            return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")

        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [i for i in ids if i not in special_ids]

        return self._tokenizer.decode(ids)

    def encode_code(
        self,
        code: str,
        language: str = "python",
        task: str = "gen",
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode code with language and task context tokens.
        Example output: [<bos>, <task:gen>, <lang:py>, ...code tokens..., <eos>]
        """
        lang_map = {
            "python": "<lang:py>", "py": "<lang:py>",
            "javascript": "<lang:js>", "js": "<lang:js>",
            "typescript": "<lang:ts>", "ts": "<lang:ts>",
            "cpp": "<lang:cpp>", "c++": "<lang:cpp>",
            "java": "<lang:java>",
            "go": "<lang:go>", "golang": "<lang:go>",
            "rust": "<lang:rs>", "rs": "<lang:rs>",
        }
        task_map = {
            "gen": "<task:gen>", "generate": "<task:gen>",
            "fix": "<task:fix>", "debug": "<task:fix>",
            "explain": "<task:explain>",
            "optimize": "<task:optimize>", "refactor": "<task:optimize>",
        }

        lang_token = lang_map.get(language.lower(), "<lang:py>")
        task_token = task_map.get(task.lower(), "<task:gen>")

        prefix = f"{task_token} {lang_token} "
        full_text = prefix + code

        return self.encode(full_text, add_special_tokens=True, max_length=max_length)

    def save(self, path: str):
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)

        if hasattr(self, "_use_hf") and self._use_hf:
            self._hf_tokenizer.save_pretrained(path)
        elif self._tokenizer is not None:
            self._tokenizer.save(os.path.join(path, "tokenizer.json"))

        # Save config
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "model_type": "shulker_bpe",
            }, f, indent=2)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]

    def __len__(self) -> int:
        return self.vocab_size
