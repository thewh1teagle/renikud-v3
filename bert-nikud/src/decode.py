"""
Utility functions for reconstructing Hebrew text with nikud from predictions.
"""

import torch


def reconstruct_text_from_predictions(
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    vowel_preds: torch.Tensor,
    dagesh_preds: torch.Tensor,
    sin_preds: torch.Tensor,
    stress_preds: torch.Tensor,
    prefix_preds: torch.Tensor,
    tokenizer,
) -> str:
    """
    Reconstruct Hebrew text with nikud marks from model predictions.

    Args:
        input_ids: Token IDs [seq_len]
        offset_mapping: Offset mapping for each token [seq_len, 2] (char_start, char_end)
        vowel_preds: Vowel class predictions [seq_len] (0-7)
        dagesh_preds: Dagesh binary predictions [seq_len] (0/1)
        sin_preds: Sin binary predictions [seq_len] (0/1)
        stress_preds: Stress binary predictions [seq_len] (0/1)
        prefix_preds: Prefix binary predictions [seq_len] (0/1)
        tokenizer: Tokenizer for decoding token IDs

    Returns:
        Text with predicted nikud marks
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

    # Get special token IDs
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    # Use offset_mapping to properly map predictions to characters
    # offset_mapping has shape [seq_len, 2] where each entry is (char_start, char_end)
    for i in range(1, len(input_ids)):
        # Stop at special tokens
        if input_ids[i].item() == sep_token_id or input_ids[i].item() == pad_token_id:
            break

        # Get character range using offset_mapping
        char_start, char_end = offset_mapping[i].tolist()
        char_start = int(char_start)
        char_end = int(char_end)

        # Get the character from input_ids
        char = tokenizer.decode([input_ids[i].item()])
        result.append(char)

        # Only add nikud marks for Hebrew letters
        if char not in LETTERS:
            continue

        # Skip nikud for final letters (they can't have nikud)
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
    text = "".join(result)
    text = normalize(text)  # Apply same normalization as training data (outputs NFD)

    return text
