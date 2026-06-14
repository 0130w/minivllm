import torch
import torch.nn as nn


@torch.compile(mode="reduce-overhead")
class Linear(nn.Module):
    """
    Linear(x) = x @ W + b
    """

    def __init__(self, in_features, out_features, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) / in_features**0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None

    def forward(self, x):
        output = x @ self.weight
        if self.bias:
            output += self.bias
        return output
