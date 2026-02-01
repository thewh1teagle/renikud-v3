"""Tokenizer loading utilities.

Root cause note (important):
The DictaBERT repos set `tokenizer_class: "BertTokenizer"` in tokenizer_config.json.
`AutoTokenizer.from_pretrained(..., use_fast=True)` will still honor that and load
*the slow BertTokenizer*, which does **word-level** tokenization and turns Hebrew
words into `[UNK]` (because the vocab is character-oriented and lacks `##` forms).

We must therefore force the *fast* tokenizer implementation backed by
`tokenizer.json`, which correctly splits into individual characters.

We still load strictly from pretrained (HF) — no local-path special casing.
"""

from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_path: str):
    # Force fast tokenizer (tokenizer.json backend) regardless of tokenizer_class.
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
