import torch
from torch import nn


class Att(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        d = q.shape[-1]
        return torch.softmax((q @ k.T) / d ** 0.5, -1) @ v


class MultiAttention(nn.Module):
    def __init__(self, head, input_dim, output_dim):
        super().__init__()
        self._head = head
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._q_map_net = nn.Linear(input_dim, input_dim * head)
        self._k_map_net = nn.Linear(input_dim, input_dim * head)
        self._v_map_net = nn.Linear(output_dim, output_dim * head)
        self._out_map_net = nn.Linear(self._head * self._output_dim, output_dim)

    def forward(self, q, k, v):
        d = q.shape[-1]
        _qs = self._q_map_net(q).reshape(-1,  q.shape[1],self._head, self._input_dim).permute(0, 2, 1, 3)
        _ks = self._k_map_net(k).reshape(-1,  k.shape[1],self._head, self._input_dim).permute(0, 2, 1, 3)
        _vs = self._v_map_net(v).reshape(-1,  v.shape[1], self._head,self._output_dim).permute(0, 2, 1, 3)
        # print(_vs.shape)

        out = torch.softmax((_qs @ _ks.transpose(-1, -2)) / d ** 0.5, -1) @ _vs
        out = out.permute(0, 2, 1, 3).reshape(-1, q.shape[1], self._head * self._output_dim)
        out = self._out_map_net(out)
        return out


if __name__ == '__main__':
    q = torch.randn(8, 3, 5)
    k = torch.randn(8, 7, 5)
    v = torch.randn(8, 7, 4)
    att = MultiAttention(2, 5, 4)
    result = att(q, k, v)
    print(result.shape)
