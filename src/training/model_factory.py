from transformers import TrainingArguments
import os
from src.config.model_config import TrainingConfig
from src.training.enhanced_trainer import EnhancedTrainer
from src.dataset.dataset_processor import DatasetProcessor
from src.models.peft_model import DoRAModelForSequenceClassification

class ModelFactory:
    @staticmethod
    def create_training_arguments(config: TrainingConfig) -> TrainingArguments:
        return TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_first_step=True,
            dataloader_num_workers=2
        )

    @staticmethod
    def create_model_and_trainer(config: TrainingConfig) -> EnhancedTrainer:
        # Prepare dataset
        dataset_processor = DatasetProcessor(config.model_name, config.max_length)
        tokenized_datasets = dataset_processor.prepare_dataset(config.dataset_name)

        # Create DoRA model
        dora_model_wrapper = DoRAModelForSequenceClassification(
            model_name=config.model_name,
            peft_config=config.peft_config,
            model_config=config.model_config,
            dora_config=config.dora_config
        )
        model = dora_model_wrapper.create_model()

        # Create training arguments
        training_args = ModelFactory.create_training_arguments(config)

        # Initialize trainer
        trainer = EnhancedTrainer(
            dora_config=config.dora_config,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
        )

        return trainer