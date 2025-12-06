# µGraphCoder: On-Device Generative Graph Neural Networks

Implementation of the µGraphCoder framework for dynamic IoT topologies, as described in the paper.

## Features

- **Topology Fingerprinting**: Compact representation of graph structures
- **Hypernetwork**: Generates model weights from fingerprints
- **Vector-Quantized Codebook**: Efficient weight storage and retrieval
- **Low-Rank Adapters**: On-device fine-tuning for concept drift
- **Hardware Simulation**: Deployment-aware training constraints
- **Synthetic Data Generation**: IoT network simulation

## Installation

### Using uv (recommended)

```bash
# Clone repository
git clone <repository-url>
cd miuGraphCoder

# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install PyTorch Geometric dependencies (additional step)
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+${cpu}.html
# Replace ${cpu} with your platform (e.g., cpu, cu118, etc.)