from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, Any

class DatasetProcessor:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_function(self, examples: Dict[str, Any]):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def prepare_dataset(self, dataset_name: str):
        dataset = load_dataset(dataset_name)
        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        return tokenized_datasets 