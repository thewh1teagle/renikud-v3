"""
Inference script for Hebrew Nikud BERT model with benchmark evaluation.

Download benchmark data first:
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv

Usage:
    uv pip install git+https://github.com/thewh1teagle/phonikud.git
    uv run scripts/infer.py --checkpoint <path> --gt gt.tsv
"""

import csv
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm
import jiwer

from phonikud import phonemize

# Import from src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import HebrewNikudModel
from decode import reconstruct_text_from_predictions
from tokenizer_utils import load_tokenizer


class NikudPredictor:
    def __init__(
        self, checkpoint_path: str, tokenizer_path: str, device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading model on {device}...")

        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model = HebrewNikudModel()

        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.is_dir():
            checkpoint_file = checkpoint_file / "model.safetensors"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        self.model.load_state_dict(load_file(checkpoint_file, device=str(device)))
        self.model.to(device)
        self.model.eval()
        print("Model loaded!")

    def predict(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text

        # Remove existing diacritics
        text = __import__("re").sub(r"[\u0590-\u05CF]", "", text)

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0]

        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask)

        return reconstruct_text_from_predictions(
            text,
            input_ids[0],
            offset_mapping,
            predictions["vowel"][0],
            predictions["dagesh"][0],
            predictions["sin"][0],
            predictions["stress"][0],
            predictions["prefix"][0],
            self.tokenizer,
        )


def load_gt(filepath: str):
    """Load ground truth TSV: Sentence, Phonemes, Field"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append(
                {
                    "sentence": row["Sentence"],
                    "phonemes": row["Phonemes"],
                    "field": row.get("Field", ""),
                }
            )
    return data


def evaluate(predictor: NikudPredictor, gt_data: list):
    """Run inference and compute WER/CER against ground truth phonemes."""
    wer_scores = []
    cer_scores = []
    examples = []

    print(f"\nEvaluating {len(gt_data)} samples...")

    for item in tqdm(gt_data):
        sentence = item["sentence"]
        gt_phonemes = item["phonemes"]

        # Predict nikud and convert to phonemes
        nikud_text = predictor.predict(sentence)
        pred_phonemes = phonemize(nikud_text)

        # Calculate metrics
        wer = jiwer.wer(gt_phonemes, pred_phonemes)
        cer = jiwer.cer(gt_phonemes, pred_phonemes)

        wer_scores.append(wer)
        cer_scores.append(cer)

        # Store first 5 examples
        if len(examples) < 5:
            examples.append(
                {
                    "sentence": sentence,
                    "gt": gt_phonemes,
                    "pred": pred_phonemes,
                    "nikud": nikud_text,
                }
            )

    mean_wer = sum(wer_scores) / len(wer_scores)
    mean_cer = sum(cer_scores) / len(cer_scores)

    # Print 5 examples
    print(f"\n{'=' * 80}")
    print("Sample Predictions (first 5):")
    print(f"{'=' * 80}")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Input:    {ex['sentence']}")
        print(f"   Nikud:    {ex['nikud']}")
        print(f"   GT:       {ex['gt']}")
        print(f"   Pred:     {ex['pred']}")

    print(f"\n{'=' * 80}")
    print("Results:")
    print(f"  Mean WER: {mean_wer:.4f}")
    print(f"  Mean CER: {mean_cer:.4f}")
    print(f"  Samples:  {len(gt_data)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hebrew Nikud model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--gt", type=str, default="gt.tsv", help="Ground truth TSV file"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="dicta-il/dictabert-large-char-menaked"
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if not Path(args.gt).exists():
        print(f"Error: {args.gt} not found. Download it with:")
        print(
            "wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv"
        )
        return

    predictor = NikudPredictor(args.checkpoint, args.tokenizer, args.device)
    gt_data = load_gt(args.gt)
    evaluate(predictor, gt_data)


if __name__ == "__main__":
    main()
