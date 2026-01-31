"""
Workflow:
1) Pre-tokenize and cache the dataset (required for training):
   uv run src/pre_tokenize.py --data_dir dataset/ --model_name ivrit-ai/whisper-large-v3-turbo --dataset_cache_path ./.dataset-cache

2) Train the model (loads from the cached dataset):
   uv run src/train.py --dataset_cache_path ./.dataset-cache --model_name ivrit-ai/whisper-large-v3-turbo --output_dir whisper-heb-nikud --batch_size 16 --learning_rate 1e-5 --max_steps 1000
   OR
   export WANDB_PROJECT=whisper-heb-nikud
   uv run src/train.py --dataset_cache_path ./.dataset-cache --model_name ivrit-ai/whisper-large-v3-turbo --output_dir whisper-heb-nikud --batch_size 16 --learning_rate 1e-5 --max_steps 90000 --report_to wandb

To upload the model to the hub:
uv run hf upload --repo-type model whisper-heb-nikud ./whisper-heb-nikud

To use with wandb:
uv run wandb login
"""

import datasets
import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Union
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from config import get_args
from data import DataCollatorSpeechSeq2SeqWithPadding
from eval import compute_metrics

args = get_args()

def main():
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(args.model_name, language="Hebrew", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Set generation config
    model.generation_config.language = "hebrew"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None # Deprecated

    # Load dataset from cache (output of src/pre_tokenize.py)
    if not Path(args.dataset_cache_path).exists():
        raise FileNotFoundError(
            f"Missing cache at {args.dataset_cache_path}. "
            "Run pre-tokenize (set your data dir): uv run src/pre_tokenize.py --data_dir dataset/ --model_name "
            f"{args.model_name} --dataset_cache_path {args.dataset_cache_path}"
        )

    dataset = datasets.load_from_disk(args.dataset_cache_path)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        # Gradient checkpointing is disabled to fix a "backward through the graph a second time" RuntimeError.
        # This error occurs when gradient checkpointing is enabled alongside a custom data collator.
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=args.report_to,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        processing_class=processor,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()
