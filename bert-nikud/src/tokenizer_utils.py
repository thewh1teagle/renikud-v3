"""Tokenizer loading utilities."""

from pathlib import Path
import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast


UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
BLANK_TOKEN = "[BLANK]"


def load_tokenizer(tokenizer_path: str):
    path = Path(tokenizer_path)
    tokenizer_json = path / "tokenizer.json"
    tokenizer_config = path / "tokenizer_config.json"

    if path.is_dir() and tokenizer_json.exists():
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            unk_token=UNK_TOKEN,
            cls_token=CLS_TOKEN,
            sep_token=SEP_TOKEN,
            pad_token=PAD_TOKEN,
            mask_token=MASK_TOKEN,
            additional_special_tokens=[BLANK_TOKEN],
        )
        if tokenizer_config.exists():
            with open(tokenizer_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_max_length = config.get("model_max_length")
            if model_max_length:
                tokenizer.model_max_length = model_max_length
        return tokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
