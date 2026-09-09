"""
Simple MLP baseline — intentionally plain. This arm exists to show what
Arms B (graph) and C (text) add ON TOP OF raw features, so don't over-tune it.
"""
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)