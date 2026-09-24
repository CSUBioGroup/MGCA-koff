import math

import torch
from torch import nn


class MultiHeadInteractionAttention(nn.Module):
    def __init__(self, heads: int = 8, conv: int = 32):
        super().__init__()
        self.heads = heads
        self.channels = conv * 3
        self.drug_projection = nn.Linear(self.channels, self.channels * heads)
        self.protein_projection = nn.Linear(self.channels, self.channels * heads)

    def forward(self, drug, protein):
        batch, channels, drug_len = drug.shape
        protein_len = protein.shape[-1]
        drug_attention = torch.relu(self.drug_projection(drug.transpose(1, 2)))
        drug_attention = drug_attention.view(batch, self.heads, drug_len, channels)
        protein_attention = torch.relu(self.protein_projection(protein.transpose(1, 2)))
        protein_attention = protein_attention.view(batch, self.heads, protein_len, channels)
        interaction = torch.matmul(drug_attention, protein_attention.transpose(2, 3))
        interaction = torch.tanh(interaction / math.sqrt(self.channels)).mean(dim=1)
        drug_weight = torch.tanh(interaction.sum(dim=2)).unsqueeze(1)
        protein_weight = torch.tanh(interaction.sum(dim=1)).unsqueeze(1)
        return drug * drug_weight, protein * protein_weight


class AttentionDTA(nn.Module):
    """Device-safe cleanup of the existing AttentionDTA architecture."""

    def __init__(self, conv: int = 32, char_dim: int = 128, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.protein_embedding = nn.Embedding(26, char_dim, padding_idx=0)
        self.drug_embedding = nn.Embedding(65, char_dim, padding_idx=0)
        self.drug_convs = nn.Sequential(
            nn.Conv1d(char_dim, conv, 4), nn.ReLU(),
            nn.Conv1d(conv, conv * 2, 6), nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 3, 8), nn.ReLU(),
        )
        self.protein_convs = nn.Sequential(
            nn.Conv1d(char_dim, conv, 4), nn.ReLU(),
            nn.Conv1d(conv, conv * 2, 8), nn.ReLU(),
            nn.Conv1d(conv * 2, conv * 3, 12), nn.ReLU(),
        )
        self.attention = MultiHeadInteractionAttention(heads=heads, conv=conv)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(conv * 6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        nn.init.constant_(self.output.bias, 5)

    def forward(self, smiles, protein):
        drug = self.drug_convs(self.drug_embedding(smiles).transpose(1, 2))
        target = self.protein_convs(self.protein_embedding(protein).transpose(1, 2))
        drug, target = self.attention(drug, target)
        pair = torch.cat((self.pool(drug).squeeze(-1), self.pool(target).squeeze(-1)), dim=1)
        pair = self.dropout(self.activation(self.fc1(pair)))
        pair = self.dropout(self.activation(self.fc2(pair)))
        pair = self.activation(self.fc3(pair))
        return self.output(pair)
