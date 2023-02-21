import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    "Position embeddings"

    def __init__(self, d_model: int, max_len: int = 5000):
        """Génère des embeddings de position

        Args:
            d_model (int): Dimension des embeddings à générer
            max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.v_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.q_linear = nn.Linear(input_dim, input_dim)

        self.layers_out = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())
    
    def forward(self, x):
        """
        x : (bs, len, dim)
        """
        log_attentions = (self.q_linear(x) @ self.k_linear(x).transpose(1, 2)) / math.sqrt(x.size(-1))  # (bs, len, len)
        attentions = torch.softmax(log_attentions, dim=-1)   # (bs, len, len)
        weighted_input = (attentions @ self.v_linear(x))     # (bs, len, dim)
        return self.layers_out(weighted_input)

class ResidualSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.v_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.q_linear = nn.Linear(input_dim, input_dim)

        self.layers_out = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())
    
    def forward(self, x):
        """
        x : (bs, len, dim)
        """
        log_attentions = (self.q_linear(x) @ self.k_linear(x).transpose(1, 2)) / math.sqrt(x.size(-1))  # (bs, len, len)
        attentions = torch.softmax(log_attentions, dim=-1)   # (bs, len, len)
        weighted_input = (attentions @ self.v_linear(x))     # (bs, len, dim)
        return self.layers_out(weighted_input + x)


class BasicAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            SelfAttention(input_dim, input_dim),
            SelfAttention(input_dim, input_dim),
            SelfAttention(input_dim, output_dim)
        )

        self.pos_emb = PositionalEncoding(input_dim)
    
    def forward(self, x):
        x = self.pos_emb(x)
        outputs = self.model(x).squeeze(-1)
        return outputs.mean(dim=-1)


class ResidualAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            SelfAttention(input_dim, input_dim),
            SelfAttention(input_dim, input_dim),
            SelfAttention(input_dim, output_dim)
        )
    
    def forward(self, x):
        outputs = self.model(x).squeeze(-1)
        return outputs.mean(dim=-1)


