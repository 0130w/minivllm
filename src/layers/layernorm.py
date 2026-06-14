import torch
import torch.nn as nn


@torch.compile(mode="reduce-overhead")
class BatchNorm1D(nn.Module):
    """
    BatchNorm1D(x) = gamma * (x - x_mean) / sqrt(x_var + eps) + beta
    Shape of x should be (B, N, C)
    """

    def __init__(self, dim, gamma=None, beta=None, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = (
            nn.Parameter(torch.ones(dim))
            if gamma is None
            else nn.Parameter(gamma.detach().clone())
        )
        self.beta = (
            nn.Parameter(torch.zeros(dim))
            if beta is None
            else nn.Parameter(beta.detach().clone())
        )
        self.running_mean = torch.ones(dim)
        self.running_var = torch.zeros(dim)

    def forward(self, x):
        if self.training:
            x_mean = x.mean((0, 1), keepdim=True)
            x_var = x.var((0, 1), keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        output = self.gamma * (x - x_mean) / torch.sqrt(x_var + self.eps) + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * x_var
        return output


# TODO: figure out what is redisual RMSNorm
@torch.compile("reduce-overhead")
class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / sqrt(x ** 2 + eps) * gamma
    """

    def __init__(self, dim, gamma=None, eps=1e-5):
        super().__init__()
        self.gamma = (
            nn.Parameter(torch.ones(dim))
            if gamma is None
            else nn.Parameter(gamma.detach().clone())
        )
        self.eps = eps

    def forward(self, x):
        output = x / (torch.sqrt(x**2 + self.eps)) * self.gamma
        return output
