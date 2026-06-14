import torch
import torch.nn as nn
from linear import Linear


@torch.compile(mode="reduce-overhead")
class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size, use_bias=False):
        super().__init__()
        self.linear = Linear(d_model, vocab_size, use_bias)

    def forward(self, x):
        return self.linear(x)
