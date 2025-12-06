import torch

class HardwareSimulator:
    """Simulate hardware constraints for deployment-aware training."""
    
    def __init__(self, target_device='cortex-m4'):
        self.target_device = target_device
        
        # Latency lookup table (simplified)
        self.latency_lut = {
            'cortex-m4': {
                'gcn_32x5': 2.5,    # ms for 32x5 matmul
                'gcn_32x32': 3.0,   # ms for 32x32 matmul
                'gcn_2x32': 1.5,    # ms for 2x32 matmul
                'relu': 0.1,
                'norm': 0.5,
                'pool': 0.3,
            }
        }
        
        # Memory costs (bytes)
        self.memory_costs = {
            'weight_32bit': 4,
            'weight_8bit': 1,
            'activation_32bit': 4,
            'activation_8bit': 1,
        }
    
    def estimate_latency(self, gnn_config, quantization_bits=8):
        """Estimate inference latency."""
        if self.target_device not in self.latency_lut:
            return 0.0
        
        lut = self.latency_lut[self.target_device]
        total_latency = 0.0
        
        # Input layer
        total_latency += lut['gcn_32x5']  # 5 features to 32 hidden
        
        # Hidden layers
        num_hidden = gnn_config.get('num_hidden_layers', 1)
        for _ in range(num_hidden):
            total_latency += lut['gcn_32x32']
            total_latency += lut['relu']
            total_latency += lut['norm']
        
        # Output layer
        total_latency += lut['gcn_2x32']
        
        # Quantization effect (simplified)
        if quantization_bits == 8:
            total_latency *= 0.7  # 30% speedup with 8-bit
        
        return total_latency
    
    def estimate_memory(self, gnn, quantization_bits=8):
        """Estimate memory usage."""
        total_memory = 0
        
        # Weights memory
        if hasattr(gnn, 'injected_weights'):
            for weight in gnn.injected_weights:
                num_params = weight.numel()
                if quantization_bits == 8:
                    total_memory += num_params * self.memory_costs['weight_8bit']
                else:
                    total_memory += num_params * self.memory_costs['weight_32bit']
        
        # Activation memory (estimated max)
        # Assuming max batch size of 1 and 32 hidden dimensions
        activation_memory = 32 * 50 * self.memory_costs['activation_32bit']  # 50 nodes max
        
        total_memory += activation_memory
        
        return total_memory  # in bytes
    
    def check_constraints(self, gnn, max_latency_ms=50, max_memory_kb=96):
        """Check if model meets hardware constraints."""
        latency = self.estimate_latency({'num_hidden_layers': gnn.num_layers - 1})
        memory = self.estimate_memory(gnn) / 1024  # Convert to KB
        
        latency_ok = latency <= max_latency_ms
        memory_ok = memory <= max_memory_kb
        
        return {
            'latency_ms': latency,
            'memory_kb': memory,
            'latency_ok': latency_ok,
            'memory_ok': memory_ok,
            'all_ok': latency_ok and memory_ok
        }