import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from src.config.model_config import ModelConfig
from src.dataset.dataset_processor import DatasetProcessor
from src.utils.model_utils import setup_model_and_tokenizer
from src.utils.metrics import compute_metrics
import os

def main():
    # Load configuration
    config = ModelConfig()
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    dataset_processor = DatasetProcessor(
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    datasets = dataset_processor.load_and_process_data("data/train.csv", split=True)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=True,  # Enable mixed precision training
        gradient_checkpointing=True,  # Enable gradient checkpointing
        save_total_limit=2,  # Keep only the last 2 checkpoints
        remove_unused_columns=False,  # Important for custom datasets
        # Update metric logging configurations
        logging_dir=f"{config.output_dir}/logs",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",  # Use log loss as the primary metric
        greater_is_better=False,  # Lower log loss is better
    )
    
    # Initialize trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,  # Now imported from metrics module
    )
    
    # Train the model
    trainer.train()
    
    # Create evaluation output directory
    eval_output_dir = os.path.join(config.output_dir, "eval_results")
    
    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset, output_dir=eval_output_dir)
    print(f"Final evaluation results: {eval_results}")
    
    # Save the final model
    trainer.save_model(f"{config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{config.output_dir}/final_model")

if __name__ == "__main__":
    main()