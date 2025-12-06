import torch
import torch.nn as nn

class LowRankAdapter(nn.Module):
    """Low-rank adapter for on-device fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_features) * 0.01)
        
        # Scaling factor
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Apply adapter to base weight."""
        adapter = self.scale * (self.A @ self.B)
        return base_weight + adapter.T  # Transpose to match weight dimensions
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return self.A.numel() + self.B.numel() + 1
    
class AdapterEnhancedGNN(nn.Module):
    """GNN enhanced with low-rank adapters."""
    
    def __init__(self, base_gnn: nn.Module, rank: int = 4):
        super().__init__()
        self.base_gnn = base_gnn
        self.rank = rank
        
        # Create adapters for each layer
        self.adapters = nn.ModuleList()
        weight_shapes = base_gnn.get_weight_shapes()
        
        for out_dim, in_dim in weight_shapes:
            self.adapters.append(LowRankAdapter(in_dim, out_dim, rank))
        
    def forward(self, x, edge_index, batch=None):
        # Apply adapters to weights
        if hasattr(self.base_gnn, 'injected_weights'):
            adapted_weights = []
            for base_weight, adapter in zip(self.base_gnn.injected_weights, self.adapters):
                adapted_weight = adapter(base_weight.squeeze(0))
                adapted_weights.append(adapted_weight.unsqueeze(0))
            
            self.base_gnn.inject_weights(adapted_weights)
        
        return self.base_gnn(x, edge_index, batch)
    
    def freeze_base(self):
        """Freeze the base GNN weights."""
        for param in self.base_gnn.parameters():
            param.requires_grad = False
    
    def unfreeze_adapters(self):
        """Unfreeze only the adapter parameters."""
        for param in self.adapters.parameters():
            param.requires_grad = True
    
    def get_adapter_parameters(self):
        """Get only adapter parameters for optimization."""
        return list(self.adapters.parameters())
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        for adapter in self.adapters:
            total += adapter.get_num_parameters()
        return total