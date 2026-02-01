"""Tokenizer loading utilities.

We intentionally *always* load tokenizers via `from_pretrained(...)` to avoid
silent local-vs-remote mismatches (special tokens, vocab, normalizers, etc.).
"""

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str):
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
