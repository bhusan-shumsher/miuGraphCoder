import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    """Hypernetwork that generates code indices from topology fingerprints."""
    
    def __init__(self, fingerprint_dim: int = 128, hidden_dim: int = 256,
                 num_codebooks: int = 8, codes_per_book: int = 256):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codes_per_book = codes_per_book
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Code predictors (one per codebook)
        self.code_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, codes_per_book)
            )
            for _ in range(num_codebooks)
        ])
        
        # Continuous code generators (for quantization)
        self.continuous_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 32)  # code_dim = 32
            )
            for _ in range(num_codebooks)
        ])
        
    def forward(self, fingerprints, temperature: float = 1.0, hard: bool = False):
        """Generate code indices from fingerprints.
        
        Args:
            fingerprints: [batch_size, fingerprint_dim]
            temperature: Softmax temperature
            hard: If True, use hard sampling
            
        Returns:
            code_indices: Discrete indices [batch_size, num_codebooks]
            logits: Raw predictions [batch_size, num_codebooks, codes_per_book]
            continuous_codes: Continuous codes for quantization [batch_size, num_codebooks, code_dim]
        """
        batch_size = fingerprints.shape[0]
        
        # Encode fingerprint
        encoded = self.encoder(fingerprints)
        
        # Generate logits for each codebook
        all_logits = []
        for predictor in self.code_predictors:
            logits = predictor(encoded)  # [batch_size, codes_per_book]
            all_logits.append(logits.unsqueeze(1))
        
        logits = torch.cat(all_logits, dim=1)  # [batch_size, num_codebooks, codes_per_book]
        
        # Generate continuous codes
        continuous_codes = []
        for generator in self.continuous_generators:
            codes = generator(encoded)  # [batch_size, code_dim]
            continuous_codes.append(codes.unsqueeze(1))
        
        continuous_codes = torch.cat(continuous_codes, dim=1)  # [batch_size, num_codebooks, code_dim]
        
        # Sample discrete indices
        if self.training or not hard:
            # Gumbel softmax sampling
            code_indices = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            # Convert one-hot to indices
            code_indices = torch.argmax(code_indices, dim=-1)
        else:
            # Hard sampling
            code_indices = torch.argmax(logits, dim=-1)
        
        return code_indices, logits, continuous_codes
    
    def generate_continuous_only(self, fingerprints):
        """Generate only continuous codes (for quantization training)."""
        encoded = self.encoder(fingerprints)
        
        continuous_codes = []
        for generator in self.continuous_generators:
            codes = generator(encoded)
            continuous_codes.append(codes.unsqueeze(1))
        
        return torch.cat(continuous_codes, dim=1)