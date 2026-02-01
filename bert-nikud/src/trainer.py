"""
Custom Trainer for Hebrew Nikud model with WER/CER metrics.
"""

import torch
from transformers import Trainer
from evaluate import calculate_wer, calculate_cer
from decode import reconstruct_text_from_predictions


class NikudTrainer(Trainer):
    """Custom Trainer for Hebrew Nikud model with WER/CER metrics."""

    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, processing_class=processing_class, **kwargs)
        self.processing_class = processing_class

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss for training."""
        # Extract all labels
        vowel_labels = inputs.pop("vowel_labels")
        dagesh_labels = inputs.pop("dagesh_labels")
        sin_labels = inputs.pop("sin_labels")
        stress_labels = inputs.pop("stress_labels")
        prefix_labels = inputs.pop("prefix_labels")

        # Remove non-model fields
        inputs.pop("plain_text", None)
        inputs.pop("original_text", None)
        inputs.pop("offset_mapping", None)

        outputs = model(
            **inputs,
            vowel_labels=vowel_labels,
            dagesh_labels=dagesh_labels,
            sin_labels=sin_labels,
            stress_labels=stress_labels,
            prefix_labels=prefix_labels,
        )
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Custom evaluation loop that includes WER/CER calculations."""
        # Call parent evaluation
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Calculate WER/CER
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        device = self.accelerator.device

        total_wer = 0.0
        total_cer = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Get predictions
                predictions = model.predict(input_ids, attention_mask)

                # Calculate WER/CER for each sample
                # Get offset mappings for this batch
                offset_mappings = batch.get("offset_mapping")

                for i in range(input_ids.shape[0]):
                    # Re-tokenize to get offset_mapping if not in batch
                    if offset_mappings is not None:
                        offset_mapping = offset_mappings[i]
                    else:
                        enc = self.processing_class(
                            batch["plain_text"][i],
                            return_tensors="pt",
                            return_offsets_mapping=True,
                            add_special_tokens=True,
                        )
                        offset_mapping = enc["offset_mapping"][0]

                    predicted_text = reconstruct_text_from_predictions(
                        batch["plain_text"][i],
                        input_ids[i],
                        offset_mapping,
                        predictions["vowel"][i],
                        predictions["dagesh"][i],
                        predictions["sin"][i],
                        predictions["stress"][i],
                        predictions["prefix"][i],
                        self.processing_class,
                    )

                    target_text = batch["original_text"][i]

                    total_wer += calculate_wer(predicted_text, target_text)
                    total_cer += calculate_cer(predicted_text, target_text)
                    num_samples += 1

        # Add WER/CER to metrics
        if num_samples > 0:
            output.metrics[f"{metric_key_prefix}_wer"] = total_wer / num_samples
            output.metrics[f"{metric_key_prefix}_cer"] = total_cer / num_samples

        return output
