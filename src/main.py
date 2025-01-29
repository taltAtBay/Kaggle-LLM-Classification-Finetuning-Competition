import logging
from src.config.model_config import TrainingConfig, DoRAConfig, PEFTConfig, ModelConfig
from src.training.model_factory import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configure DoRA
    dora_config = DoRAConfig(
        rank=8,
        init_scale=0.01,
        alpha=0.2,
        beta=0.1
    )

    # Configure PEFT settings
    peft_config = PEFTConfig(
        use_8bit_quantization=True,
        use_gradient_checkpointing=True
    )

    # Configure model architecture
    model_config = ModelConfig(
        num_labels=3,  # For 3-class classification
        dropout_rate=0.1
    )

    # Configure training
    training_config = TrainingConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        dataset_name="your_dataset_name",
        output_dir="./results",
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5,
        dora_config=dora_config,
        peft_config=peft_config,
        model_config=model_config
    )

    # Create trainer
    trainer = ModelFactory.create_model_and_trainer(training_config)

    # Train the model
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    logger.info(f"Evaluation Results: {results}")

if __name__ == "__main__":
    main() 