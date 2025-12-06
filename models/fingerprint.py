import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigsh
import networkx as nx

class TopologyFingerprintExtractor:
    """Extract compact topology fingerprints from graphs."""
    
    def __init__(self, fingerprint_dim: int = 128):
        self.fingerprint_dim = fingerprint_dim
        
    def extract(self, data) -> torch.Tensor:
        """Extract fingerprint from PyG data object."""
        # Convert to networkx for graph metrics
        G = self._pyg_to_nx(data)
        
        fingerprint_parts = []
        
        # 1. Basic graph statistics
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1) // 2 if data.edge_index is not None else 0
        
        fingerprint_parts.extend([
            num_nodes / 100.0,  # Normalized
            num_edges / 500.0,
            nx.density(G) if num_nodes > 1 else 0.0
        ])
        
        # 2. Degree statistics
        if num_nodes > 0:
            degrees = [d for _, d in G.degree()]
            fingerprint_parts.extend([
                np.mean(degrees) / 10.0,
                np.std(degrees) / 10.0,
                np.max(degrees) / 10.0 if degrees else 0.0,
                np.percentile(degrees, 90) / 10.0 if len(degrees) > 1 else 0.0
            ])
        else:
            fingerprint_parts.extend([0.0] * 4)
        
        # 3. Clustering and centrality (simplified)
        try:
            if num_nodes > 2:
                clustering = nx.average_clustering(G)
                fingerprint_parts.append(clustering)
            else:
                fingerprint_parts.append(0.0)
        except:
            fingerprint_parts.append(0.0)
        
        # 4. Spectral features (simplified approximation)
        spectral_features = self._compute_spectral_features(G, k=5)
        fingerprint_parts.extend(spectral_features)
        
        # 5. Node feature statistics
        if data.x is not None:
            node_features = data.x.numpy()
            for i in range(min(5, node_features.shape[1])):
                fingerprint_parts.append(np.mean(node_features[:, i]))
                fingerprint_parts.append(np.std(node_features[:, i]))
        
        # 6. Edge feature statistics
        if data.edge_attr is not None and len(data.edge_attr) > 0:
            edge_features = data.edge_attr.numpy()
            for i in range(min(3, edge_features.shape[1])):
                fingerprint_parts.append(np.mean(edge_features[:, i]))
                fingerprint_parts.append(np.std(edge_features[:, i]))
        
        # Pad or truncate to target dimension
        if len(fingerprint_parts) > self.fingerprint_dim:
            fingerprint = fingerprint_parts[:self.fingerprint_dim]
        else:
            fingerprint = fingerprint_parts + [0.0] * (self.fingerprint_dim - len(fingerprint_parts))
        
        return torch.FloatTensor(fingerprint).unsqueeze(0)
    
    def _pyg_to_nx(self, data):
        """Convert PyG data to networkx graph."""
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges
        if data.edge_index is not None:
            edge_index = data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i], edge_index[1, i]
                G.add_edge(u, v)
        
        return G
    
    def _compute_spectral_features(self, G, k=5):
        """Compute simplified spectral features."""
        if len(G) == 0:
            return [0.0] * k
        
        try:
            # Use normalized Laplacian eigenvalues
            L = nx.normalized_laplacian_matrix(G)
            if L.shape[0] < k:
                eigenvalues = np.zeros(k)
                eigenvalues[:L.shape[0]] = np.linalg.eigvals(L.toarray())[:L.shape[0]].real
            else:
                # Compute only k smallest eigenvalues
                eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
            
            # Normalize
            eigenvalues = np.sort(eigenvalues)
            if len(eigenvalues) < k:
                eigenvalues = np.pad(eigenvalues, (0, k - len(eigenvalues)))
            
            return eigenvalues.tolist()
        except:
            return [0.0] * k