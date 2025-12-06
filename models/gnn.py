import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TinyGNN(nn.Module):
    """Tiny GNN for on-device inference."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, 
                 output_dim: int = 2, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create placeholder layers (weights will be injected)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim, bias=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim, affine=False))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, bias=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, affine=False))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim, bias=False))
        
        # Classifier
        self.classifier = nn.Linear(output_dim, 2)
        
    def forward(self, x, edge_index, batch=None):
        # Inject weights if they were provided
        if hasattr(self, 'injected_weights'):
            for i, (conv, weight) in enumerate(zip(self.convs, self.injected_weights)):
                conv.lin.weight = nn.Parameter(weight.squeeze(0))  # Remove batch dimension
        
        # Forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Final layer
        if self.num_layers >= 1:
            x = self.convs[-1](x, edge_index)
        
        # Pool if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def inject_weights(self, weights):
        """Inject generated weights into the GNN."""
        self.injected_weights = weights
        
    def get_weight_shapes(self):
        """Get the shapes of weight matrices needed."""
        shapes = []
        # First layer
        shapes.append((self.hidden_dim, self.input_dim))
        # Hidden layers
        for _ in range(self.num_layers - 2):
            shapes.append((self.hidden_dim, self.hidden_dim))
        # Output layer
        if self.num_layers > 1:
            shapes.append((2, self.hidden_dim))
        return shapes
    
    def quantize_weights(self, bits: int = 8):
        """Quantize weights to lower precision (simulated)."""
        if not hasattr(self, 'injected_weights'):
            return
        
        quantized_weights = []
        for weight in self.injected_weights:
            # Simple linear quantization
            w = weight.data
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / (2**bits - 1)
            zero_point = -w_min / scale
            
            w_quant = torch.round(w / scale + zero_point)
            w_dequant = (w_quant - zero_point) * scale
            
            quantized_weights.append(w_dequant)
        
        self.injected_weights = quantized_weights