import torch
from torch import nn


class ArcLoss(nn.Module):

    def __init__(self, input_dim, output_dim, m):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim, output_dim))
        self._m = m

    def forward(self, f, t):
        angle = (f * self._w).sum(-1) / (torch.norm(f, dim=-1) * torch.norm(self._w, dim=-1))
        angle = torch.arccos(angle)

        f_i = f[t == 1]
        angle_i = angle[t == 1]

        f_j = f[t == 0]
        angle_j = angle[t == 0]

        num = torch.exp(torch.norm(f_i, dim=-1) * torch.cos(angle_i + self._m))
        deno = (torch.exp(torch.norm(f_j, dim=-1) * torch.cos(angle_j))).sum(-1)

        return num / (deno + num)