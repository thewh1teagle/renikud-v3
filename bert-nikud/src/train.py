"""
Training script for Hebrew Nikud BERT model with HuggingFace Trainer.

Pretokenize first:
    uv run python src/pretokenize.py --dataset-dir dataset

Train:
    uv run accelerate launch src/train.py \
        --dataset-dir dataset \
        --eval-split val \
        --batch-size 16 \
        --max-epochs 999999 \
        --lr 1e-4 \
        --checkpoint-dir checkpoints/run_5m \
        --wandb-mode online \
        --wandb-project renikud-v2 \
        --wandb-run-name run_5m

Resume from checkpoint:
    uv run accelerate launch src/train.py --resume checkpoints/run_5m/checkpoint-6000
"""

import torch
from transformers import TrainingArguments
from pathlib import Path

from model import HebrewNikudModel, count_parameters
from dataset import load_pretokenized, collate_fn
from trainer import NikudTrainer
from config import get_args
from tokenizer_utils import load_tokenizer


def main():
    """Main training function with HuggingFace Trainer."""
    config = get_args()

    print("=" * 80)
    print("Hebrew Nikud Training")
    print("=" * 80)
    print("Configuration:")
    for key, value in vars(config).items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer and data...")
    tokenizer = load_tokenizer(config.tokenizer_path)

    # Load pretokenized datasets
    train_dataset = load_pretokenized(config.dataset_dir, config.train_split)
    eval_dataset = load_pretokenized(config.dataset_dir, config.eval_split)

    # Initialize model
    print("\nInitializing model...")
    model = HebrewNikudModel(model_name=config.model_name, dropout=config.dropout)
    total_params, trainable_params = count_parameters(model)
    print(
        f"Model: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)"
    )

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=config.max_grad_norm,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=config.save_best,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_dir=f"{config.checkpoint_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb" if config.wandb_mode != "disabled" else "none",
        seed=config.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Create trainer
    print(f"\nStarting training for {config.max_epochs} epochs...")
    print("=" * 80)
    trainer = NikudTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    # Train (with optional resume)
    resume_checkpoint = None
    if config.resume == "auto":
        resume_checkpoint = True
    elif config.resume:
        resume_checkpoint = config.resume

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)

    # Print final summary
    best_metrics = trainer.state.best_metric
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Final model: {final_path}")
    if best_metrics is not None:
        print(f"Best CER: {best_metrics:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
