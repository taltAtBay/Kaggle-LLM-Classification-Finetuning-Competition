import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.model_config import ModelConfig
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

def setup_tokenizer(config: ModelConfig):
    """Initialize tokenizer with padding token"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_model(config: ModelConfig):
    """Initialize model with configuration"""
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=getattr(torch, config.torch_dtype),
        use_flash_attention=config.use_flash_attention,
        device_map="auto"  # Automatically handle model parallelism
    )
    return model

def setup_model_and_tokenizer(config: ModelConfig):
    """Initialize model and tokenizer"""
    tokenizer = setup_tokenizer(config)
    model = setup_model(config)
    return model, tokenizer

def visualize_eval_results(eval_results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualize evaluation metrics using plots.
    
    Args:
        eval_results: Dictionary containing evaluation metrics
        output_dir: Directory to save the visualization plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    plt.figure(figsize=(12, 6))
    
    # Plot metrics as bar chart
    metrics = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.xticks(rotation=45)
    plt.title('Evaluation Metrics')
    
    # If there's a confusion matrix in the results, plot it
    if 'confusion_matrix' in eval_results:
        plt.subplot(1, 2, 2)
        sns.heatmap(eval_results['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   cmap='Blues')
        plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eval_metrics.png')
    plt.close() 