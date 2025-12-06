import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
# Remove the ".." and use sys.path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fingerprint import TopologyFingerprintExtractor
from models.hypernetwork import HyperNetwork
from models.codebook import WeightCodebook
from models.gnn import TinyGNN
from models.adapter import AdapterEnhancedGNN

# from ..models.fingerprint import TopologyFingerprintExtractor
# from ..models.hypernetwork import HyperNetwork
# from ..models.codebook import WeightCodebook
# from ..models.gnn import TinyGNN
# from ..models.adapter import AdapterEnhancedGNN

class MiuGraphCoderTrainer:
    """Trainer for the complete µGraphCoder system."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.fingerprint_extractor = TopologyFingerprintExtractor(fingerprint_dim=128)
        self.hypernetwork = HyperNetwork(
            fingerprint_dim=128,
            hidden_dim=256,
            num_codebooks=8,
            codes_per_book=256
        ).to(self.device)
        
        self.codebook = WeightCodebook(
            num_codebooks=8,
            codes_per_book=256,
            code_dim=32,
            output_dims=[5, 32, 2]  # For 2-layer GNN
        ).to(self.device)
        
        self.gnn_template = TinyGNN(input_dim=5, hidden_dim=32, output_dim=2, num_layers=2)
        
        # Optimizers
        self.hyper_optimizer = optim.AdamW(
            list(self.hypernetwork.parameters()) + list(self.codebook.parameters()),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # Loss functions
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.vq_loss_weight = 0.1
        
        # Metrics
        self.train_losses = []
        self.val_accuracies = []
        
    def train_step(self, graphs, labels):
        """Single training step."""
        self.hypernetwork.train()
        self.codebook.train()
        
        # Extract fingerprints
        fingerprints = []
        for graph in graphs:
            fp = self.fingerprint_extractor.extract(graph).to(self.device)
            fingerprints.append(fp)
        
        fingerprints = torch.cat(fingerprints, dim=0)
        
        # Generate codes
        code_indices, logits, continuous_codes = self.hypernetwork(
            fingerprints, temperature=1.0, hard=False
        )
        
        # Quantize and get VQ loss
        vq_indices, vq_loss, perplexity = self.codebook.quantize_and_encode(continuous_codes)
        
        # Generate weights from indices
        weights = self.codebook(vq_indices)
        
        # Task loss
        task_loss = 0
        total_correct = 0
        total_samples = 0
        
        for i, graph in enumerate(graphs):
            # Create GNN with generated weights
            gnn = TinyGNN(input_dim=5, hidden_dim=32, output_dim=2, num_layers=2).to(self.device)
            layer_weights = [w[i:i+1] for w in weights]  # Get batch element
            gnn.inject_weights(layer_weights)
            
            # Forward pass
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            
            output = gnn(x, edge_index, batch)
            
            # Compute loss
            target = torch.tensor([labels[i]], dtype=torch.long, device=self.device)
            task_loss += self.task_loss_fn(output, target)
            
            # Accuracy
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += 1
        
        task_loss = task_loss / len(graphs)
        accuracy = total_correct / total_samples
        
        # Total loss
        total_loss = task_loss + self.vq_loss_weight * vq_loss
        
        # Backward pass
        self.hyper_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.hypernetwork.parameters()) + list(self.codebook.parameters()),
            max_norm=1.0
        )
        self.hyper_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': perplexity.item(),
            'accuracy': accuracy
        }
    
    def train_adapter(self, base_gnn, graphs, labels, num_epochs=10):
        """Train low-rank adapter on device."""
        adapter_gnn = AdapterEnhancedGNN(base_gnn, rank=4).to(self.device)
        adapter_gnn.freeze_base()
        adapter_gnn.unfreeze_adapters()
        
        optimizer = optim.Adam(adapter_gnn.get_adapter_parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_correct = 0
            
            for graph, label in zip(graphs, labels):
                x = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                target = torch.tensor([label], dtype=torch.long, device=self.device)
                
                output = adapter_gnn(x, edge_index, batch)
                loss = self.task_loss_fn(output, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
            
            avg_loss = total_loss / len(graphs)
            accuracy = total_correct / len(graphs)
            
            print(f"Adapter Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return adapter_gnn
    
    def evaluate(self, test_graphs, test_labels):
        """Evaluate the system on test data."""
        self.hypernetwork.eval()
        self.codebook.eval()
        
        with torch.no_grad():
            # Extract fingerprints
            fingerprints = []
            for graph in test_graphs:
                fp = self.fingerprint_extractor.extract(graph).to(self.device)
                fingerprints.append(fp)
            
            fingerprints = torch.cat(fingerprints, dim=0)
            
            # Generate codes (hard sampling for evaluation)
            code_indices, _, _ = self.hypernetwork(
                fingerprints, temperature=0.5, hard=True
            )
            
            # Generate weights
            weights = self.codebook(code_indices)
            
            # Evaluate each graph
            total_correct = 0
            predictions = []
            
            for i, graph in enumerate(test_graphs):
                gnn = TinyGNN(input_dim=5, hidden_dim=32, output_dim=2, num_layers=2).to(self.device)
                layer_weights = [w[i:i+1] for w in weights]
                gnn.inject_weights(layer_weights)
                
                x = graph.x.to(self.device)
                edge_index = graph.edge_index.to(self.device)
                batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                
                output = gnn(x, edge_index, batch)
                pred = output.argmax(dim=1)
                
                predictions.append(pred.item())
                
                if pred.item() == test_labels[i]:
                    total_correct += 1
        
        accuracy = total_correct / len(test_graphs)
        return accuracy, predictions
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'hypernetwork_state_dict': self.hypernetwork.state_dict(),
            'codebook_state_dict': self.codebook.state_dict(),
            'optimizer_state_dict': self.hyper_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
        self.codebook.load_state_dict(checkpoint['codebook_state_dict'])
        self.hyper_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"Checkpoint loaded from {path}")