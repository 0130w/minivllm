import torch
import torch.nn as nn
import time


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sigmoid()

    def forward(self, x):
        return self.layer(x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.SiLU()

    @torch.compile(mode="reduce-overhead")
    def forward(self, x):
        return self.layer(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.GELU()

    @torch.compile(mode="redue-overhead")
    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    layer = SiLU()
    input_tensor = torch.ones(2, 3)

    for _ in range(10):
        _ = layer(input_tensor)

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        output = layer(input_tensor)
        end_time = time.time()
        torch.cuda.synchronize()
        times.append(end_time - start_time)
