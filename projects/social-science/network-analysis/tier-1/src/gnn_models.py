"""
Graph Neural Network models for social network analysis.

Implements GraphSAGE, GAT, and Temporal GNN architectures.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, GATConv, GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch and PyTorch Geometric not available. Install with: pip install torch torch-geometric")


if TORCH_AVAILABLE:
    class GraphSAGE(nn.Module):
        """
        GraphSAGE model for node embedding and influence prediction.

        Architecture:
        - 2-3 GraphSAGE convolutional layers
        - ReLU activations
        - Dropout for regularization
        - Final embedding dimension configurable
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 256,
            out_channels: int = 128,
            num_layers: int = 2,
            dropout: float = 0.5
        ):
            super(GraphSAGE, self).__init__()

            self.num_layers = num_layers
            self.dropout = dropout

            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

            self.convs.append(SAGEConv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x


    class GAT(nn.Module):
        """
        Graph Attention Network for learning node importance.

        Uses multi-head attention to weight neighbor contributions.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 256,
            out_channels: int = 128,
            num_layers: int = 2,
            heads: int = 8,
            dropout: float = 0.5
        ):
            super(GAT, self).__init__()

            self.num_layers = num_layers
            self.dropout = dropout

            self.convs = nn.ModuleList()

            # First layer
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            )

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
                )

            # Output layer
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
            )

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
            return x


    class TemporalGNN(nn.Module):
        """
        Temporal Graph Neural Network for capturing network dynamics.

        Combines GNN with temporal attention mechanism.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 256,
            out_channels: int = 128,
            num_layers: int = 2,
            dropout: float = 0.5
        ):
            super(TemporalGNN, self).__init__()

            self.num_layers = num_layers
            self.dropout = dropout

            # Spatial convolutions
            self.spatial_convs = nn.ModuleList()
            self.spatial_convs.append(GCNConv(in_channels, hidden_channels))

            for _ in range(num_layers - 2):
                self.spatial_convs.append(GCNConv(hidden_channels, hidden_channels))

            self.spatial_convs.append(GCNConv(hidden_channels, out_channels))

            # Temporal attention
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=4,
                dropout=dropout
            )

        def forward(self, x, edge_index, temporal_snapshots=None):
            """
            Forward pass with optional temporal snapshots.

            Parameters:
            -----------
            x : Tensor
                Node features [num_nodes, in_channels]
            edge_index : Tensor
                Edge indices [2, num_edges]
            temporal_snapshots : list of Tensors, optional
                List of node embeddings from previous time steps
            """
            # Spatial convolution
            for i, conv in enumerate(self.spatial_convs):
                x = conv(x, edge_index)
                if i < len(self.spatial_convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

            # Temporal attention (if snapshots provided)
            if temporal_snapshots is not None:
                # Stack current and past snapshots
                # Shape: [num_snapshots, num_nodes, out_channels]
                temporal_seq = torch.stack(temporal_snapshots + [x], dim=0)

                # Apply temporal attention
                x_attended, _ = self.temporal_attention(
                    temporal_seq[-1:],  # Query: current snapshot
                    temporal_seq,       # Key/Value: all snapshots
                    temporal_seq
                )
                x = x_attended.squeeze(0)

            return x


    class InfluencePredictionHead(nn.Module):
        """
        Prediction head for influence score prediction.

        Takes node embeddings and predicts influence scores.
        """

        def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int = 128,
            dropout: float = 0.3
        ):
            super(InfluencePredictionHead, self).__init__()

            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 1)
            self.dropout = dropout

        def forward(self, embeddings):
            """
            Predict influence scores from node embeddings.

            Parameters:
            -----------
            embeddings : Tensor
                Node embeddings [num_nodes, embedding_dim]

            Returns:
            --------
            Tensor : Influence scores [num_nodes, 1]
            """
            x = F.relu(self.fc1(embeddings))
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.fc3(x)
            return x.squeeze(-1)


    def train_gnn_model(
        model,
        data,
        optimizer,
        criterion,
        device='cpu'
    ):
        """
        Single training step for GNN model.

        Parameters:
        -----------
        model : nn.Module
            GNN model
        data : torch_geometric.data.Data
            Graph data
        optimizer : torch.optim.Optimizer
            Optimizer
        criterion : nn.Module
            Loss function
        device : str
            Device to use

        Returns:
        --------
        float : Training loss
        """
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x.to(device), data.edge_index.to(device))

        # Compute loss (assuming node classification task)
        if hasattr(data, 'y') and hasattr(data, 'train_mask'):
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        else:
            # If no labels, use reconstruction loss
            loss = F.mse_loss(out, data.x.to(device))

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()


    @torch.no_grad()
    def evaluate_gnn_model(
        model,
        data,
        criterion,
        device='cpu'
    ):
        """
        Evaluate GNN model.

        Returns:
        --------
        float : Validation/test loss
        """
        model.eval()

        out = model(data.x.to(device), data.edge_index.to(device))

        if hasattr(data, 'y') and hasattr(data, 'val_mask'):
            loss = criterion(out[data.val_mask], data.y[data.val_mask])
        else:
            loss = F.mse_loss(out, data.x.to(device))

        return loss.item()

else:
    # Placeholder classes if PyTorch not available
    class GraphSAGE:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch torch-geometric")

    class GAT:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch torch-geometric")

    class TemporalGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch torch-geometric")
