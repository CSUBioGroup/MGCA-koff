import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_max_pool


class GraphDTA(nn.Module):
    """GCNNet variant selected by the existing GraphDTA re-implementation."""

    def __init__(self, n_filters: int = 32, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        atom_features = 78
        output_dim = 128
        self.conv1 = GCNConv(atom_features, atom_features)
        self.conv2 = GCNConv(atom_features, atom_features * 2)
        self.conv3 = GCNConv(atom_features * 2, atom_features * 4)
        self.graph_fc1 = nn.Linear(atom_features * 4, 1024)
        self.graph_fc2 = nn.Linear(1024, output_dim)
        self.n_filters = n_filters
        self.protein_embedding = nn.Embedding(26, embed_dim)
        # This orientation intentionally matches the published GraphDTA code.
        self.protein_conv = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.protein_fc = nn.Linear(n_filters * 121, output_dim)
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        graph = F.relu(self.conv1(data.x, data.edge_index))
        graph = F.relu(self.conv2(graph, data.edge_index))
        graph = F.relu(self.conv3(graph, data.edge_index))
        graph = global_max_pool(graph, data.batch)
        graph = self.dropout(F.relu(self.graph_fc1(graph)))
        graph = self.dropout(self.graph_fc2(graph))

        protein = self.protein_embedding(data.target)
        protein = self.protein_conv(protein).reshape(-1, self.n_filters * 121)
        protein = self.protein_fc(protein)

        pair = torch.cat((graph, protein), dim=1)
        pair = self.dropout(F.relu(self.fc1(pair)))
        pair = self.dropout(F.relu(self.fc2(pair)))
        return self.output(pair)
