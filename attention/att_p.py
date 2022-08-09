import torch
from torch import nn


class Att(nn.Module):
    def __init__(self):
        super(Att, self).__init__()

    def forward(self, q, k, v):
        d = q.shape[-1]
        return torch.softmax((q @ k.T) / d ** 0.5, -1) @ v


if __name__ == '__main__':
    q = torch.randn(3, 5)
    k = torch.randn(7, 5)
    v = torch.randn(7, 5)
    att = Att()
    result = att(q, k, v)
    print(result.shape)
