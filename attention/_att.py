import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        d = K.shape[-1]
        return torch.softmax((Q @ K.T) / d ** 0.5, -1) @ V


class MultiAttention(nn.Module):

    def __init__(self, head, input_dim):
        super().__init__()
        self._head = head
        self._input_dim = input_dim

        self._q_map_net = nn.Linear(input_dim, input_dim * head)
        self._k_map_net = nn.Linear(input_dim, input_dim * head)
        self._v_map_net = nn.Linear(input_dim, input_dim * head)

        self._output_net = nn.Linear(input_dim * head, input_dim)

    def forward(self, q, k, v):
        _qs = self._q_map_net(q)
        _ks = self._k_map_net(k)
        _vs = self._v_map_net(v)

        _qs = _qs.reshape(-1, self._head, self._input_dim).permute(1, 0, 2)
        _ks = _ks.reshape(-1, self._head, self._input_dim).permute(1, 0, 2)
        _vs = _vs.reshape(-1, self._head, self._input_dim).permute(1, 0, 2)

        _as = torch.softmax((_qs @ _ks.permute(0, 2, 1)) / self._input_dim ** 0.5, -1) @ _vs

        _vs = _as.permute(1, 0, 2).reshape(-1, self._head * self._input_dim)

        return self._output_net(_vs)


if __name__ == '__main__':
    q = torch.randn(3, 5)
    k = torch.randn(7, 5)
    v = torch.randn(7, 5)

    att = MultiAttention(2, 5)
    att = Attention()
    y = att(q, k, v)
    print(y.shape)
