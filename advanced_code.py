import os
import logging
from typing import Optional
from dataclasses import dataclass
from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning settings"""
    model_name: str
    data_path: str
    username: str
    token: str
    project_name: str
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-5
    gradient_accumulation: int = 8
    text_column: str = "text"
    train_split: str = "train"
    push_to_hub: bool = True

class LLMFineTuner:
    """Handler for LLM fine-tuning using AutoTrain Advanced"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate the configuration settings"""
        if self.config.push_to_hub and (not self.config.username or not self.config.token):
            raise ValueError("Username and token are required when push_to_hub is True")
        
        if not os.path.exists(self.config.data_path) and not self.config.data_path.startswith(("http", "https")):
            raise ValueError(f"Data path does not exist: {self.config.data_path}")
    
    def _create_training_params(self) -> LLMTrainingParams:
        """Create AutoTrain training parameters"""
        logger.info("Setting up training parameters...")
        
        return LLMTrainingParams(
            model=self.config.model_name,
            data_path=self.config.data_path,
            chat_template="tokenizer",
            text_column=self.config.text_column,
            train_split=self.config.train_split,
            trainer="sft",
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            peft=True,
            quantization="int4",
            target_modules="all-linear",
            padding="right",
            optimizer="paged_adamw_8bit",
            scheduler="cosine",
            gradient_accumulation=self.config.gradient_accumulation,
            mixed_precision="bf16",
            merge_adapter=True,
            project_name=self.config.project_name,
            log="tensorboard",
            push_to_hub=self.config.push_to_hub,
            username=self.config.username,
            token=self.config.token,
        )
    
    def train(self) -> None:
        """Execute the fine-tuning process"""
        try:
            logger.info(f"Starting fine-tuning for model: {self.config.model_name}")
            
            # Create training parameters
            params = self._create_training_params()
            
            # Initialize and create project
            logger.info("Initializing AutoTrain project...")
            project = AutoTrainProject(
                params=params,
                backend="local",
                process=True
            )
            
            logger.info("Starting training process...")
            project.create()
            
            logger.info("Training completed successfully!")
            
            if self.config.push_to_hub:
                logger.info(f"Model pushed to HuggingFace Hub: {self.config.username}/{self.config.project_name}")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Configure your environment variables
        hf_username = os.getenv("HF_USERNAME")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_username or not hf_token:
            logger.warning("HF_USERNAME or HF_TOKEN not found in environment variables")
        
        # Create configuration
        config = FineTuningConfig(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            data_path="HuggingFaceH4/no_robots",
            username=hf_username,
            token=hf_token,
            project_name="autotrain-llama32-1b-finetune",
            epochs=3,
            batch_size=1,
            learning_rate=1e-5,
            gradient_accumulation=8
        )
        
        # Initialize and run fine-tuning
        fine_tuner = LLMFineTuner(config)
        fine_tuner.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()