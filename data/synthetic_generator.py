import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import List, Tuple
import random

class SyntheticIoTGraphGenerator:
    """Generate synthetic IoT network graphs for training and testing."""
    
    def __init__(self, num_devices_range: Tuple[int, int] = (10, 50),
                 anomaly_prob: float = 0.1):
        self.num_devices_range = num_devices_range
        self.anomaly_prob = anomaly_prob
        
    def generate_graph(self, time_step: int = 0, drift_level: float = 0.0) -> Data:
        """Generate a single temporal graph with optional drift."""
        num_devices = np.random.randint(*self.num_devices_range)
        
        # Base graph structure (preferential attachment for IoT-like networks)
        G = nx.barabasi_albert_graph(num_devices, m=2)
        
        # Add drift: randomly add/remove edges and nodes
        if drift_level > 0:
            G = self._apply_drift(G, drift_level)
        
        # Generate node features (simulated IoT device characteristics)
        node_features = self._generate_node_features(G, time_step)
        
        # Generate edge features (communication patterns)
        edge_index, edge_features = self._generate_edge_features(G)
        
        # Generate labels (anomaly detection task)
        y = self._generate_labels(G, node_features, edge_features)
        
        return Data(
            x=torch.FloatTensor(node_features),
            edge_index=edge_index,
            edge_attr=torch.FloatTensor(edge_features) if edge_features is not None else None,
            y=torch.LongTensor(y),
            num_nodes=num_devices
        )
    
    def _apply_drift(self, G: nx.Graph, drift_level: float) -> nx.Graph:
        """Apply topology drift to the graph."""
        G = G.copy()
        
        # Randomly remove some edges
        edges_to_remove = random.sample(list(G.edges()), 
                                        k=int(len(G.edges()) * drift_level * 0.3))
        G.remove_edges_from(edges_to_remove)
        
        # Add some new edges (device joins/communication changes)
        nodes = list(G.nodes())
        for _ in range(int(len(G.edges()) * drift_level * 0.4)):
            if len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
        
        # Add/remove nodes (device churn)
        if random.random() < drift_level * 0.3:
            # Add new node
            new_node = max(nodes) + 1 if nodes else 0
            G.add_node(new_node)
            # Connect to existing nodes
            connections = random.sample(nodes, min(3, len(nodes)))
            for node in connections:
                G.add_edge(new_node, node)
        
        return G
    
    def _generate_node_features(self, G: nx.Graph, time_step: int) -> np.ndarray:
        """Generate node features for IoT devices."""
        num_nodes = len(G.nodes())
        
        # Feature dimensions: [cpu_usage, memory_usage, packet_rate, device_type, uptime]
        features = np.zeros((num_nodes, 5))
        
        for i in range(num_nodes):
            features[i, 0] = np.random.beta(2, 5)  # CPU usage (low for most devices)
            features[i, 1] = np.random.beta(3, 3)  # Memory usage
            features[i, 2] = np.random.exponential(0.5)  # Packet rate
            features[i, 3] = np.random.choice([0, 1, 2])  # Device type (0:sensor, 1:gateway, 2:actuator)
            features[i, 4] = np.random.exponential(10) + time_step * 0.1  # Uptime
        
        # Add time-based patterns
        features[:, 0] += 0.1 * np.sin(time_step * 0.1 + np.arange(num_nodes) * 0.01)
        
        return features
    
    def _generate_edge_features(self, G: nx.Graph) -> Tuple[torch.Tensor, np.ndarray]:
        """Generate edge indices and features."""
        edge_index = []
        edge_features = []
        
        for u, v in G.edges():
            edge_index.append([u, v])
            # Bidirectional communication
            edge_index.append([v, u])
            
            # Edge features: [traffic_volume, latency, protocol_type]
            traffic = np.random.exponential(1.0)
            latency = np.random.exponential(0.1)
            protocol = np.random.choice([0, 1, 2])  # 0:MQTT, 1:CoAP, 2:HTTP
            
            edge_features.append([traffic, latency, protocol])
            # Symmetric features for reverse direction
            edge_features.append([traffic * 0.9, latency, protocol])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = np.array(edge_features) if edge_features else None
        
        return edge_index, edge_features
    
    def _generate_labels(self, G: nx.Graph, node_features: np.ndarray, 
                        edge_features: np.ndarray) -> List[int]:
        """Generate anomaly labels for nodes."""
        num_nodes = len(G.nodes())
        labels = [0] * num_nodes  # 0: normal, 1: anomaly
        
        # Determine anomalies based on features
        for i in range(num_nodes):
            anomaly_score = 0
            
            # High CPU usage anomaly
            if node_features[i, 0] > 0.8:
                anomaly_score += 1
            
            # Unusual packet rate
            if node_features[i, 2] > 2.0:
                anomaly_score += 1
            
            # Check neighbor anomalies (propagation)
            neighbors = list(G.neighbors(i))
            if neighbors:
                neighbor_anomaly = any(labels[j] == 1 for j in neighbors[:3])
                if neighbor_anomaly and random.random() < 0.3:
                    anomaly_score += 1
            
            # Random anomaly injection
            if random.random() < self.anomaly_prob:
                anomaly_score = 2
            
            labels[i] = 1 if anomaly_score >= 2 else 0
        
        return labels
    
    def generate_temporal_sequence(self, num_steps: int = 100, 
                                   drift_schedule: List[float] = None) -> List[Data]:
        """Generate a sequence of temporal graphs with drift."""
        if drift_schedule is None:
            drift_schedule = [0.0] + [0.1 * (i/num_steps) for i in range(1, num_steps)]
        
        graphs = []
        for t in range(num_steps):
            drift = drift_schedule[t] if t < len(drift_schedule) else drift_schedule[-1]
            graphs.append(self.generate_graph(t, drift))
        
        return graphs