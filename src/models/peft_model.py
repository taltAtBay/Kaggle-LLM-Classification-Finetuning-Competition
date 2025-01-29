import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from src.config.model_config import PEFTConfig, ModelConfig, DoRAConfig
from src.models.dora_adapter import DoRAAdapter

class DoRAModelForSequenceClassification:
    def __init__(self, model_name: str, peft_config: PEFTConfig, model_config: ModelConfig, dora_config: DoRAConfig):
        self.model_name = model_name
        self.peft_config = peft_config
        self.model_config = model_config
        self.dora_config = dora_config

    def create_model(self):
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.model_config.num_labels,
            load_in_8bit=self.peft_config.use_8bit_quantization,
            device_map="auto"
        )

        if self.peft_config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Configure target modules if not specified
        if self.peft_config.target_modules is None:
            self.peft_config.target_modules = self._get_default_target_modules()

        # Apply DoRA to specified layers
        adapted_params = 0
        for name, module in model.named_modules():
            if any(target in name for target in self.peft_config.target_modules):
                if isinstance(module, nn.Linear):
                    adapter = DoRAAdapter(module.weight.data, self.dora_config)
                    setattr(module, 'dora_adapter', adapter)
                    adapted_params += adapter.U.numel() + adapter.V.numel()

        print(f"Applied DoRA to {adapted_params} parameters")
        return model

    def _get_default_target_modules(self):
        """Get default target modules for LLaMA architecture"""
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"] 