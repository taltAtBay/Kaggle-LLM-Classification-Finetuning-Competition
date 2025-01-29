import torch
import torch.nn as nn
from src.config.model_config import DoRAConfig

class DoRAAdapter(nn.Module):
    """Implements DoRA adaptation layer"""
    def __init__(self, weight: torch.Tensor, config: DoRAConfig):
        super().__init__()
        u, s, v = torch.svd(weight)
        rank = min(config.rank, s.size(0))
        
        self.U = nn.Parameter(torch.randn(weight.size(0), rank) * config.init_scale)
        self.V = nn.Parameter(torch.randn(rank, weight.size(1)) * config.init_scale)
        self.register_buffer('original_weight', weight.detach())
        self.alpha = config.alpha
        self.beta = config.beta

    def forward(self):
        adaptation = torch.mm(self.U, self.V)
        return self.original_weight + self.alpha * adaptation * torch.norm(self.original_weight) / torch.norm(adaptation) 