import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v):
        super().__init__()
        self.W_q = torch.randn(d_model, d_q)
        self.W_k = torch.randn()
        self.W_v = torch.randn()

    def forward(self, x):
        pass
