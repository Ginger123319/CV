import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, cls_num, feature_dim):
        super().__init__()

        self._center = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, f, t):
        _t = t.long()
        _c = self._center[_t]
        # print(_c.shape)
        # print(f.shape)

        _d = torch.sum((f - _c) ** 2, dim=-1) ** 0.5
        _h = torch.histc(t.float(), 10, min=0, max=9)
        _n = _h[_t]
        return torch.sum(_d / _n)
