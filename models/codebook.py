import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector quantization layer for weight codebook."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        # inputs: [batch_size, num_vectors, embedding_dim]
        batch_size, num_vectors, _ = inputs.shape
        
        # Flatten
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(batch_size, num_vectors, self.embedding_dim)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity (for monitoring)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(batch_size, num_vectors)
    
    def get_codebook_entries(self, indices):
        """Retrieve codebook entries for given indices."""
        # indices: [batch_size, num_vectors]
        return self.embedding(indices)

class WeightCodebook(nn.Module):
    """Complete weight codebook system for GNN weight generation."""
    
    def __init__(self, num_codebooks: int = 8, codes_per_book: int = 256, 
                 code_dim: int = 32, output_dims: list = None):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codes_per_book = codes_per_book
        self.code_dim = code_dim
        
        # Default output dimensions for a small GNN
        if output_dims is None:
            # [input_dim, hidden_dim, output_dim] for 2 layers
            output_dims = [16, 32, 2]
        
        self.output_dims = output_dims
        
        # Create codebooks for each weight matrix
        self.codebooks = nn.ModuleList([
            VectorQuantizer(codes_per_book, code_dim)
            for _ in range(num_codebooks)
        ])
        
        # Mapping from codebook indices to actual weight dimensions
        self.weight_maps = nn.ModuleList()
        total_params = 0
        
        # For each GNN layer, create mapping from codebook space to weight space
        for i in range(len(output_dims) - 1):
            in_dim = output_dims[i]
            out_dim = output_dims[i + 1]
            
            # Weight matrix size
            weight_size = in_dim * out_dim
            total_params += weight_size
            
            # Create linear projection from codebook output to weight matrix
            self.weight_maps.append(
                nn.Linear(code_dim * 2, weight_size)  # Using 2 codebooks per weight matrix
            )
        
        self.total_params = total_params
        
    def forward(self, code_indices):
        """Generate weights from code indices.
        
        Args:
            code_indices: [batch_size, num_codebooks]
            
        Returns:
            List of weight matrices for GNN layers
        """
        batch_size = code_indices.shape[0]
        weights = []
        
        # For each layer, combine outputs from 2 codebooks
        layer_idx = 0
        for i in range(0, self.num_codebooks, 2):
            if i + 1 >= self.num_codebooks:
                break
                
            # Get codebook entries
            codes1 = self.codebooks[i].get_codebook_entries(code_indices[:, i])
            codes2 = self.codebooks[i+1].get_codebook_entries(code_indices[:, i+1])
            
            # Concatenate and map to weight space
            combined = torch.cat([codes1, codes2], dim=-1)
            weight_flat = self.weight_maps[layer_idx](combined)
            
            # Reshape to proper weight matrix dimensions
            in_dim = self.output_dims[layer_idx]
            out_dim = self.output_dims[layer_idx + 1]
            weight_matrix = weight_flat.view(batch_size, out_dim, in_dim)
            
            weights.append(weight_matrix)
            layer_idx += 1
            
            if layer_idx >= len(self.weight_maps):
                break
        
        return weights
    
    def quantize_and_encode(self, continuous_codes):
        """Quantize continuous codes and return indices and quantization loss."""
        batch_size, num_codebooks, code_dim = continuous_codes.shape
        
        all_indices = []
        total_vq_loss = 0
        perplexities = []
        
        for i in range(num_codebooks):
            # Reshape for VQ: [batch_size, 1, code_dim]
            codes_i = continuous_codes[:, i:i+1, :]
            
            # Quantize
            quantized, vq_loss, perplexity, indices = self.codebooks[i](codes_i)
            
            all_indices.append(indices)
            total_vq_loss += vq_loss
            perplexities.append(perplexity)
        
        indices = torch.cat(all_indices, dim=1)  # [batch_size, num_codebooks]
        avg_perplexity = torch.mean(torch.stack(perplexities))
        
        return indices, total_vq_loss, avg_perplexity