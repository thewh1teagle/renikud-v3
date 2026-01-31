import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import pandas as pd
from datasets import Audio, Dataset, DatasetDict
import torch

from config import SEP


def load_dataset_from_csv(
    data_dir: str,
    sampling_rate: int = 16000,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "metadata.csv", sep=SEP, header=None, names=["filename", "text"])

    audio_paths = [str(data_path / "wav" / f"{filename}.wav") for filename in df["filename"]]
    dataset = Dataset.from_dict({"audio": audio_paths, "text": df["text"].tolist()})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset.train_test_split(test_size=test_size, seed=seed)


def prepare_dataset(batch: Dict[str, List], processor):
    audio = batch["audio"]
    arrays = [a["array"] for a in audio]
    sampling_rate = audio[0]["sampling_rate"] if audio else 16000

    features = processor.feature_extractor(arrays, sampling_rate=sampling_rate)
    labels = processor.tokenizer(batch["text"])

    return {
        "input_features": features["input_features"],
        "labels": labels["input_ids"],
    }


def preprocess_dataset(
    dataset: DatasetDict,
    processor,
    num_proc: int | None = None,
    batch_size: int = 16,
) -> DatasetDict:
    if num_proc is None:
        num_proc = min(6, os.cpu_count() or 1)

    return dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
