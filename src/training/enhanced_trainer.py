import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict
import logging
from src.config.model_config import DoRAConfig

logger = logging.getLogger(__name__)

class EnhancedTrainer(Trainer):
    """Enhanced trainer with DoRA support and improved logging"""
    def __init__(self, dora_config: DoRAConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dora_config = dora_config
        self.best_eval_loss = float('inf')

    def apply_dora(self):
        """Apply DoRA to model weights"""
        adapted_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                adapter = DoRAAdapter(module.weight.data, self.dora_config)
                setattr(module, 'dora_adapter', adapter)
                adapted_params += adapter.U.numel() + adapter.V.numel()
        
        logger.info(f"Applied DoRA to {adapted_params} parameters")

    def training_step(self, model, inputs):
        """Override training step to include DoRA updates"""
        for module in model.modules():
            if hasattr(module, 'dora_adapter'):
                module.weight.data = module.dora_adapter()
        
        return super().training_step(model, inputs)

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with memory usage and gradient stats"""
        if torch.cuda.is_available():
            logs['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
        
        if hasattr(self, 'model') and self.model.training:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            logs['gradient_norm'] = total_norm ** 0.5
        
        super().log(logs) 