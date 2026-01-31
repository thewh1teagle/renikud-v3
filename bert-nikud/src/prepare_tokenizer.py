#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers", "transformers"]
# ///
"""Build a char-level tokenizer from bert-nikud dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import unicodedata
import re
import string
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    decoders,
    processors,
    Regex,
)
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


HEBREW_DIACRITICS = re.compile(r"[\u0590-\u05CF]")
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
SPECIAL_TOKEN_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    str(BASE_DIR / "dataset" / "train.txt"),
    str(BASE_DIR / "dataset" / "val.txt"),
]
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "tokenizer" / "char_tokenizer")
MODEL_MAX_LENGTH = 2048
EXTRA_VOCAB_CHARS = set(string.digits + string.punctuation)


def normalize_text(text: str, strip_nikud: bool) -> str:
    text = unicodedata.normalize("NFD", text)
    if strip_nikud:
        text = HEBREW_DIACRITICS.sub("", text)
    return text


def iter_texts(files: list[Path], strip_nikud: bool):
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if not text:
                    continue
                yield normalize_text(text, strip_nikud)


def build_vocab(files: list[Path], strip_nikud: bool):
    chars: set[str] = set(EXTRA_VOCAB_CHARS)
    for text in iter_texts(files, strip_nikud):
        chars.update(text)
    return sorted(chars)


def build_tokenizer(chars: list[str]):
    vocab = dict(SPECIAL_TOKEN_IDS)
    for char in chars:
        if char not in vocab:
            vocab[char] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"."), behavior="isolated"
    )
    tokenizer.decoder = decoders.Fuse()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
        pair=f"{CLS_TOKEN} $A {SEP_TOKEN} $B {SEP_TOKEN}",
        special_tokens=[
            (CLS_TOKEN, vocab[CLS_TOKEN]),
            (SEP_TOKEN, vocab[SEP_TOKEN]),
        ],
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Build char-level tokenizer")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--keep-nikud",
        action="store_true",
        help="Keep nikud marks in the vocabulary",
    )
    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chars = build_vocab(files, strip_nikud=not args.keep_nikud)
    tokenizer = build_tokenizer(chars)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
        model_max_length=MODEL_MAX_LENGTH,
    )
    fast.model_max_length = MODEL_MAX_LENGTH

    fast.save_pretrained(output_dir)
    print(f"Saved tokenizer to: {output_dir}")
    print(f"Vocab size: {len(fast)}")

    sample = "האיש רצה"
    encoding = fast(
        sample,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    print(f"Sample: {sample}")
    print(f"Tokens: {encoding.tokens()}")
    print(f"Input IDs: {encoding['input_ids']}")
    print(f"Offsets: {encoding['offset_mapping']}")


if __name__ == "__main__":
    main()
