import torch
from torch import nn


class CBow(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个随机值作为词向量,此处是10个词语，每个词语用一个长度为3的矢量表示
        self._emb = nn.Parameter(torch.randn(5, 10, 3))
        # print(self._emb)
        self._layer = nn.Linear(12, 3)

    def forward(self, x):
        # 此处x是词向量的索引
        # 取出来的形状就是SV
        vec = self._emb[x]
        # 输入全连接之前需要reshape成一个矢量
        vec = vec.reshape(-1, 12)
        print(vec.shape)
        vec = self._layer(vec)
        return vec


if __name__ == '__main__':
    net = CBow()
    print(net([0, 1, 3, 4]))
    print("new!")
