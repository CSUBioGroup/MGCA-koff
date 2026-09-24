import torch
from torch import nn
from torch.nn import functional as F


class DeepDTA(nn.Module):
    """DeepDTA architecture used by the existing re-implementation."""

    def __init__(self, num_filters: int = 32, dropout: float = 0.1):
        super().__init__()
        # Keep the original table sizes for parameter-count compatibility.
        self.smi_embedding = nn.Embedding(100, 128)
        self.smi_convs = nn.Sequential(
            nn.Conv1d(128, num_filters, 4), nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, 4), nn.ReLU(),
            nn.Conv1d(num_filters * 2, num_filters * 3, 4), nn.ReLU(),
        )
        self.protein_embedding = nn.Embedding(1000, 128)
        self.protein_convs = nn.Sequential(
            nn.Conv1d(128, num_filters, 8), nn.ReLU(),
            nn.Conv1d(num_filters, num_filters * 2, 8), nn.ReLU(),
            nn.Conv1d(num_filters * 2, num_filters * 3, 8), nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters * 6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles, protein):
        drug = self.pool(self.smi_convs(self.smi_embedding(smiles).transpose(1, 2))).squeeze(-1)
        target = self.pool(self.protein_convs(self.protein_embedding(protein).transpose(1, 2))).squeeze(-1)
        hidden = torch.cat((drug, target), dim=-1)
        hidden = self.dropout(F.relu(self.fc1(hidden)))
        hidden = self.dropout(F.relu(self.fc2(hidden)))
        hidden = F.relu(self.fc3(hidden))
        return self.output(hidden)
