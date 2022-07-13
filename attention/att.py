import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim

    def forward(self, Q, K, V):
        print(torch.softmax((Q @ K.transpose(-1, -2)) / self._input_dim ** 0.5, -1).shape)
        print(torch.sum(torch.softmax((Q @ K.transpose(-1, -2)) / self._input_dim ** 0.5, -1), dim=-1).shape)
        return torch.softmax((Q @ K.transpose(-1, -2)) / self._input_dim ** 0.5, -1) @ V


class MultiAttention(nn.Module):

    def __init__(self, head, input_dim, output_dim):
        super().__init__()
        self._head = head
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._q_map_net = nn.Linear(input_dim, input_dim * head)
        self._k_map_net = nn.Linear(input_dim, input_dim * head)
        self._v_map_net = nn.Linear(output_dim, output_dim * head)

        self._att = Attention(self._input_dim)

        self._output_net = nn.Linear(output_dim * head, output_dim)

    def forward(self, q, k, v):
        _qs = self._q_map_net(q)
        _ks = self._k_map_net(k)
        _vs = self._v_map_net(v)

        # print("1", _qs.shape, _ks.shape, _vs.shape)

        _qs = _qs.reshape(-1, q.shape[1], self._head, self._input_dim).permute(0, 2, 1, 3)
        _ks = _ks.reshape(-1, k.shape[1], self._head, self._input_dim).permute(0, 2, 1, 3)
        _vs = _vs.reshape(-1, v.shape[1], self._head, self._output_dim).permute(0, 2, 1, 3)
        # print("2", _qs.shape, _ks.shape, _vs.shape)

        _as = self._att(_qs, _ks, _vs).permute(0, 2, 1, 3)
        # print("3", _as.shape)

        _as = _as.reshape(-1, q.shape[1], self._head * self._output_dim)
        # print("4", _as.shape)

        return self._output_net(_as)


if __name__ == '__main__':
    # q：解码器输出形状为9，3，5：批次为9，输出的序列长度为3个词，每个词由长度为5的向量编码；
    # 我们需要找出在编码器输出中和这每个词最相关的输出向量，本来是需要循环3次相关性计算过程
    # k：编码器输出形状：9，7，5；批次为9，输出序列长度为7个词，每个词由长度为5的向量编码
    # 这7个词向量就是和解码器输出每个词做相关性计算的对象，每个词需要计算7次，因此需要循环3*7=21次
    # 做softmax也是在7所在的维度做，在看ppt的时候注意它是展开后的RNN，每一个输出只是一个词，也就是长度为5的向量；
    q = torch.randn(9, 3, 5)
    k = torch.randn(9, 7, 5)
    v = torch.randn(9, 7, 5)

    multi_att = Attention(5)
    y = multi_att(q, k, v)
    print(y.shape)
