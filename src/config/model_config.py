from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DoRAConfig:
    """Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation)"""
    rank: int = 8
    init_scale: float = 0.01
    alpha: float = 0.2
    beta: float = 0.1

@dataclass
class PEFTConfig:
    """Configuration for Parameter Efficient Fine-Tuning with DoRA"""
    target_modules: List[str] = None
    use_gradient_checkpointing: bool = True
    use_8bit_quantization: bool = True

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    max_length: int = 8192  # Increased to utilize more of the context window
    batch_size: int = 2     # Reduced due to longer sequences
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "outputs"
    
    # Training specific
    gradient_accumulation_steps: int = 4  # Increased for memory efficiency
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Model specific
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    model_name: str
    dataset_name: str
    output_dir: str
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dora_config: DoRAConfig = None
    peft_config: PEFTConfig = None
    model_config: ModelConfig = None 