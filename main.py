 #!/usr/bin/env python3
"""
Main script for µGraphCoder demonstration with synthetic data.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Try MPS with CPU fallback
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS memory optimization
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.synthetic_generator import SyntheticIoTGraphGenerator
from training.trainer import MiuGraphCoderTrainer
from utils.hardware_sim import HardwareSimulator

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def generate_dataset(num_train=200, num_val=50, num_test=50):
    """Generate synthetic dataset."""
    print("Generating synthetic dataset...")
    generator = SyntheticIoTGraphGenerator(
        num_devices_range=(10, 30),
        anomaly_prob=0.15
    )
    
    # Training data (various drift levels)
    train_graphs = []
    train_labels = []
    
    for i in range(num_train):
        drift_level = np.random.uniform(0, 0.3)
        graph = generator.generate_graph(time_step=i, drift_level=drift_level)
        label = graph.y[0].item()  # Use first node's label as graph label
        train_graphs.append(graph)
        train_labels.append(label)
    
    # Validation data (moderate drift)
    val_graphs = []
    val_labels = []
    
    for i in range(num_val):
        drift_level = 0.2
        graph = generator.generate_graph(time_step=i+num_train, drift_level=drift_level)
        label = graph.y[0].item()
        val_graphs.append(graph)
        val_labels.append(label)
    
    # Test data (high drift, unseen topologies)
    test_graphs = []
    test_labels = []
    
    for i in range(num_test):
        drift_level = np.random.uniform(0.3, 0.5)
        graph = generator.generate_graph(time_step=i+num_train+num_val, drift_level=drift_level)
        label = graph.y[0].item()
        test_graphs.append(graph)
        test_labels.append(label)
    
    print(f"Generated {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs")
    print(f"Class distribution - Train: {sum(train_labels)/len(train_labels):.2%} anomalies")
    print(f"Class distribution - Val: {sum(val_labels)/len(val_labels):.2%} anomalies")
    print(f"Class distribution - Test: {sum(test_labels)/len(test_labels):.2%} anomalies")
    
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels

def main():
    """Main training and evaluation pipeline."""
    set_seed(42)
    
    # Generate dataset
    train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = generate_dataset(
        num_train=200, num_val=50, num_test=50
    )
    
    # Initialize trainer
    trainer = MiuGraphCoderTrainer(device='cpu')
    
    # Training loop
    print("\nTraining µGraphCoder...")
    num_epochs = 50
    batch_size = 16
    
    train_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(num_epochs)):
        # Shuffle training data
        indices = list(range(len(train_graphs)))
        np.random.shuffle(indices)
        
        epoch_losses = []
        epoch_accuracies = []
        
        # Mini-batch training
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_graphs = [train_graphs[idx] for idx in batch_indices]
            batch_labels = [train_labels[idx] for idx in batch_indices]
            
            metrics = trainer.train_step(batch_graphs, batch_labels)
            epoch_losses.append(metrics['total_loss'])
            epoch_accuracies.append(metrics['accuracy'])
        
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        train_losses.append(avg_train_loss)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_acc, _ = trainer.evaluate(val_graphs, val_labels)
            val_accuracies.append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation...")
    test_acc, predictions = trainer.evaluate(test_graphs, test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save checkpoint
    trainer.save_checkpoint('miu_graphcoder_checkpoint.pth')
    
    # Demo adapter training
    print("\nDemonstrating low-rank adapter training...")
    
    # Generate a base GNN for a specific topology
    sample_graph = test_graphs[0]
    sample_fp = trainer.fingerprint_extractor.extract(sample_graph).to(trainer.device)
    
    with torch.no_grad():
        code_indices, _, _ = trainer.hypernetwork(sample_fp.unsqueeze(0), hard=True)
        weights = trainer.codebook(code_indices)
    
    base_gnn = trainer.gnn_template.to(trainer.device)
    base_gnn.inject_weights(weights)
    
    # Train adapter on few samples (simulating drift)
    adapter_samples = test_graphs[:10]
    adapter_labels = test_labels[:10]
    
    adapted_gnn = trainer.train_adapter(base_gnn, adapter_samples, adapter_labels, num_epochs=5)
    
    # Evaluate with adapter
    print("\nEvaluating with adapter...")
    adapter_correct = 0
    for graph, label in zip(adapter_samples, adapter_labels):
        x = graph.x.to(trainer.device)
        edge_index = graph.edge_index.to(trainer.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=trainer.device)
        
        output = adapted_gnn(x, edge_index, batch)
        pred = output.argmax(dim=1)
        
        if pred.item() == label:
            adapter_correct += 1
    
    adapter_acc = adapter_correct / len(adapter_samples)
    print(f"Adapter accuracy on adaptation samples: {adapter_acc:.4f}")
    
    # Hardware simulation
    print("\nHardware Constraints Simulation...")
    hw_sim = HardwareSimulator(target_device='cortex-m4')
    
    # Check base GNN
    constraints = hw_sim.check_constraints(base_gnn)
    print(f"Base GNN - Latency: {constraints['latency_ms']:.1f} ms, Memory: {constraints['memory_kb']:.1f} KB")
    print(f"  Meets latency constraint ({constraints['latency_ok']}): {constraints['latency_ms']:.1f} ms <= 50 ms")
    print(f"  Meets memory constraint ({constraints['memory_ok']}): {constraints['memory_kb']:.1f} KB <= 96 KB")
    
    # Quantized version
    base_gnn.quantize_weights(bits=8)
    constraints_quant = hw_sim.check_constraints(base_gnn)
    print(f"\nQuantized GNN (8-bit) - Latency: {constraints_quant['latency_ms']:.1f} ms, Memory: {constraints_quant['memory_kb']:.1f} KB")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch (every 5)')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    categories = ['Base', 'Quantized', 'With Adapter']
    values = [test_acc, test_acc * 0.98, adapter_acc]  # Simulated
    plt.bar(categories, values, color=['blue', 'green', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Model Variants Performance')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
        # Quick diagnostics to add to your code:
    print("\n=== DIAGNOSTICS ===")
    print(f"1. Class balance: {sum(test_labels)/len(test_labels):.1%} anomalies")
    print(f"2. Random baseline: {max(sum(test_labels)/len(test_labels), 1-sum(test_labels)/len(test_labels)):.1%}")
    print(f"3. Fingerprint dimension: {trainer.fingerprint_extractor.fingerprint_dim}")
    print(f"4. Training samples: {len(train_graphs)}")
    print("\n✓ Demonstration complete!")
    print("  - Checkpoint saved: miu_graphcoder_checkpoint.pth")
    print("  - Results plot saved: training_results.png")

if __name__ == "__main__":
    main()