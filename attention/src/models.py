import torch.nn.functional as F
import torch
import torch.nn as nn


class BasicAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embeddings):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.Sequential(self.linear1, self.activation, self.linear2)
        self.embeddings = embeddings

    def forward(self, x):
        emb = self.embeddings[x]
        xhat = emb.mean(axis=0)
        xhat = torch.Tensor(xhat)
        yhat = self.layers(xhat)
        return yhat


class SimpleAttention(nn.Module):
    def __init__(self, emb, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.mlp = nn.Sequential(self.linear1, self.activation, self.linear2)
        self.q = nn.Parameter(torch.randn(1, input_dim))
        self.embeddings = emb

    def forward(self, x):
        embeds = torch.from_numpy(self.embeddings[x])
        log_attentions = (self.q * embeds).sum(dim=1)
        attentions = torch.softmax(log_attentions, dim=0)
        weighted_input = (embeds * attentions.unsqueeze(-1)).sum(dim=0)
        return self.mlp(weighted_input.float())

class SelfAttention(nn.Module):
    def __init__(self, emb, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embeddings = emb

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.mlp = nn.Sequential(self.linear1, self.activation, self.linear2)

        self.q_linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        embeds = torch.from_numpy(self.embeddings[x])
        mean_input = embeds.mean(dim=0)
        q = self.q_linear(mean_input.float())
        log_attentions = (q * embeds).sum(dim=1)
        attentions = torch.softmax(log_attentions, dim=0)
        weighted_input = (embeds * attentions.unsqueeze(-1)).sum(dim=0)
        return self.mlp(weighted_input.float())