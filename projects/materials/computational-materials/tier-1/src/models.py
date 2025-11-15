"""
Graph Neural Network models for materials property prediction.

Implementations of CGCNN, ALIGNN-style, and MEGNet-style architectures.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, GATConv, NNConv, global_mean_pool


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.

    Based on: Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks
    for an Accurate and Interpretable Prediction of Material Properties"
    """

    def __init__(
        self,
        node_features: int = 16,
        edge_features: int = 8,
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_features, hidden_dim)

        self.conv_layers = nn.ModuleList(
            [CGConv(hidden_dim, edge_features) for _ in range(num_conv_layers)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_conv_layers)]
        )

        # Prediction heads
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_band_gap = nn.Linear(32, 1)
        self.fc_formation = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data, predict_property: str = "band_gap"):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # Graph convolutions with skip connections
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Skip connection

        # Global pooling
        x = global_mean_pool(x, batch)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Property-specific head
        if predict_property == "band_gap":
            out = self.fc_band_gap(x)
        elif predict_property == "formation_energy":
            out = self.fc_formation(x)
        else:
            raise ValueError(f"Unknown property: {predict_property}")

        return out.squeeze()


class ALIGNNStyle(nn.Module):
    """
    ALIGNN-style Graph Neural Network.

    Inspired by: Choudhary & DeCost (2021) "Atomistic Line Graph Neural Network
    for improved materials property predictions"

    Simplified version using graph attention.
    """

    def __init__(
        self,
        node_features: int = 16,
        edge_features: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)

        self.gat_layers = nn.ModuleList(
            [
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Prediction heads
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_band_gap = nn.Linear(32, 1)
        self.fc_formation = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data, predict_property: str = "band_gap"):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # Graph attention layers
        for gat, bn in zip(self.gat_layers, self.batch_norms):
            x_new = gat(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Skip connection

        # Global pooling
        x = global_mean_pool(x, batch)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Property-specific head
        if predict_property == "band_gap":
            out = self.fc_band_gap(x)
        elif predict_property == "formation_energy":
            out = self.fc_formation(x)
        else:
            raise ValueError(f"Unknown property: {predict_property}")

        return out.squeeze()


class MEGNetStyle(nn.Module):
    """
    MEGNet-style Graph Neural Network.

    Inspired by: Chen et al. (2019) "Graph Networks as a Universal Machine
    Learning Framework for Molecules and Crystals"
    """

    def __init__(
        self,
        node_features: int = 16,
        edge_features: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)

        # Edge network (for NNConv)
        self.edge_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.conv_layers = nn.ModuleList(
            [NNConv(hidden_dim, hidden_dim, self.edge_networks[i]) for i in range(num_layers)]
        )

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Prediction heads
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_band_gap = nn.Linear(32, 1)
        self.fc_formation = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data, predict_property: str = "band_gap"):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        edge_attr = self.edge_embedding(edge_attr)
        edge_attr = F.relu(edge_attr)

        # Message passing layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Skip connection

        # Global pooling
        x = global_mean_pool(x, batch)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Property-specific head
        if predict_property == "band_gap":
            out = self.fc_band_gap(x)
        elif predict_property == "formation_energy":
            out = self.fc_formation(x)
        else:
            raise ValueError(f"Unknown property: {predict_property}")

        return out.squeeze()


def create_model(
    model_type: str = "cgcnn",
    node_features: int = 16,
    edge_features: int = 8,
    hidden_dim: int = 128,
    **kwargs,
):
    """
    Factory function to create models.

    Args:
        model_type: Type of model ("cgcnn", "alignn", "megnet")
        node_features: Dimension of node features
        edge_features: Dimension of edge features
        hidden_dim: Hidden dimension
        **kwargs: Additional model-specific parameters

    Returns:
        PyTorch model
    """
    model_type = model_type.lower()

    if model_type == "cgcnn":
        return CGCNN(node_features, edge_features, hidden_dim, **kwargs)
    elif model_type == "alignn":
        return ALIGNNStyle(node_features, edge_features, hidden_dim, **kwargs)
    elif model_type == "megnet":
        return MEGNetStyle(node_features, edge_features, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
