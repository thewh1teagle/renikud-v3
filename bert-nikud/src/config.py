"""
Configuration module for Hebrew Nikud training.

Provides argparse setup for training parameters.
"""

import argparse


def get_args():
    """Parse command-line arguments and return configuration namespace."""
    parser = argparse.ArgumentParser(
        description="Train Hebrew Nikud BERT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory with pretokenized datasets (run pretokenize.py first)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Name of training split in .cache/ (e.g. train)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        help="Name of eval split in .cache/ (e.g. val)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps for linear schedule",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="dicta-il/dictabert-large-char",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="dicta-il/dictabert-large-char-menaked",
        help="Tokenizer path or name",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=True,
        help="Save best model based on validation metrics",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Resume from checkpoint (path or "auto" for latest)',
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cpu/cuda/mps, None for auto)",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb logging mode",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="hebrew-nikud", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name"
    )

    args = parser.parse_args()

    # Alias for convenience
    args.learning_rate = args.lr

    return args
