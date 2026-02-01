"""
Utility functions for reconstructing Hebrew text with nikud from predictions.

Key rule: reconstruction must be anchored to the *original input text*, not to
`tokenizer.decode(token_id)`. Decoding token IDs can yield "[UNK]" when the
wrong tokenizer is used (or special tokens differ), which corrupts output.

We therefore use `offset_mapping` to slice from the original `text`.
"""

import torch


def reconstruct_text_from_predictions(
    text: str,
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    vowel_preds: torch.Tensor,
    dagesh_preds: torch.Tensor,
    sin_preds: torch.Tensor,
    stress_preds: torch.Tensor,
    prefix_preds: torch.Tensor,
    tokenizer,
) -> str:
    """Reconstruct text with predicted nikud marks.

    Args:
        text: The original (diacritics-stripped) input text that was tokenized.
        input_ids: Token IDs [seq_len]
        offset_mapping: Offset mapping for each token [seq_len, 2] (char_start, char_end)
        vowel_preds: Vowel class predictions [seq_len] (0-7)
        dagesh_preds: Dagesh binary predictions [seq_len] (0/1)
        sin_preds: Sin binary predictions [seq_len] (0/1)
        stress_preds: Stress binary predictions [seq_len] (0/1)
        prefix_preds: Prefix binary predictions [seq_len] (0/1)
        tokenizer: Tokenizer (used only for special token IDs)

    Returns:
        Text with predicted nikud marks.
    """
    # Import here to avoid circular dependencies
    from dataset import ID_TO_VOWEL
    from constants import (
        DAGESH,
        S_SIN,
        STRESS_HATAMA,
        PREFIX_SEP,
        CAN_HAVE_DAGESH,
        CAN_HAVE_SIN,
        LETTERS,
        CAN_NOT_HAVE_NIKUD,
    )
    from normalize import normalize

    result = []

    # Special token IDs (may be None for some tokenizers)
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    # Start from 1 to skip [CLS] (assumes add_special_tokens=True)
    for i in range(1, len(input_ids)):
        token_id = input_ids[i].item()

        # Stop at special tokens
        if (sep_token_id is not None and token_id == sep_token_id) or (
            pad_token_id is not None and token_id == pad_token_id
        ):
            break

        char_start, char_end = offset_mapping[i].tolist()
        char_start = int(char_start)
        char_end = int(char_end)

        # Some tokenizers produce (0, 0) offsets for special tokens; skip.
        if char_end <= char_start:
            continue

        segment = text[char_start:char_end]
        result.append(segment)

        # We only know how to attach diacritics when the segment is a single Hebrew letter.
        if len(segment) != 1:
            continue

        char = segment

        if char not in LETTERS:
            continue

        if char in CAN_NOT_HAVE_NIKUD:
            continue

        diacritics = []

        # Add vowel (if not VOWEL_NONE)
        vowel_id = vowel_preds[i].item()
        if vowel_id > 0:
            vowel_char = ID_TO_VOWEL.get(vowel_id)
            if vowel_char:
                diacritics.append(vowel_char)

        # Add dagesh if predicted and character supports it
        if dagesh_preds[i].item() == 1 and char in CAN_HAVE_DAGESH:
            diacritics.append(DAGESH)

        # Add sin if predicted and character supports it
        if sin_preds[i].item() == 1 and char in CAN_HAVE_SIN:
            diacritics.append(S_SIN)

        # Add stress if predicted
        if stress_preds[i].item() == 1:
            diacritics.append(STRESS_HATAMA)

        # Sort diacritics for canonical order
        diacritics.sort()
        result.extend(diacritics)

        # Add prefix separator if predicted
        if prefix_preds[i].item() == 1:
            result.append(PREFIX_SEP)

    # Combine and normalize (keep as NFD to match training data)
    out = "".join(result)
    out = normalize(out)
    return out
