import torch
from torch import nn


class FocalBCE(nn.Module):
    # alpha调控样本均衡,beta控制样本的训练权重,困难样本的权重加大
    def __init__(self, alpha, beta):
        super().__init__()

        self._alpha = alpha
        self._beta = beta

    def forward(self, y, t):
        _p_p = y[t == 1]

        _n_p = y[t == 0]

        _p_loss = -self._alpha * (1 - _p_p) ** self._beta * torch.log(_p_p)
        _n_loss = -(1 - self._alpha) * _n_p ** self._beta * torch.log(1 - _n_p)

        # _loss = torch.mean(_p_loss) + torch.mean(_n_loss)
        _loss = torch.mean(torch.cat([_p_loss, _n_loss], dim=-1))
        return _loss


class FocalCE(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()

        self._alpha = alpha
        self._beta = beta

    def forward(self, x, t):
        _p_p = x[t == 1]

        _loss = -torch.mean(self._alpha * (1 - _p_p) ** self._beta * torch.log(_p_p))
        return _loss
