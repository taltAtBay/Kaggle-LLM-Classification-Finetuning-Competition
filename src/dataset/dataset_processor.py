import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import torch
from transformers import AutoTokenizer

class DatasetProcessor:
    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-hf", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.separator = "\n\n###\n\n"
        
    def load_and_split_data(self, csv_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the CSV file and split it into train and validation sets"""
        df = pd.read_csv(csv_path)
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, val_df
    
    def construct_prompt(self, row: pd.Series) -> str:
        """Construct the prompt template for a single row"""
        prompt_template = f"{row['prompt']}{self.separator}"
        prompt_template += f"{row['response_a']}{self.separator}"
        prompt_template += f"{row['response_b']}{self.separator}"
        prompt_template += "Choose which model response is preferred:"
        return prompt_template
    
    def get_label(self, row: pd.Series) -> int:
        """Convert the winner information to a label"""
        if row['winner_tie'] == 1:
            return 2  # Tie
        return 1 if row['winner_model_b'] == 1 else 0  # 0 for model_a, 1 for model_b
    
    def process_dataset(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Process the dataset and return tokenized inputs and labels"""
        prompts = df.apply(self.construct_prompt, axis=1).tolist()
        labels = df.apply(self.get_label, axis=1).tolist()
        
        # Tokenize the prompts
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids,
            "attention_mask": encodings.attention_mask,
            "labels": torch.tensor(labels)
        } 