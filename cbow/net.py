import torch
from torch import nn


class CBowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个随机值作为词向量,此处是10个词语，每个词语用一个长度为3的矢量表示
        self._emb = nn.Parameter(torch.randn(40, 3))
        # print(self._emb)
        self._layer = nn.Sequential(
            nn.Linear(4 * 3, 1 * 3),
            nn.Softmax(dim=-1)
        )

    def get_emb(self):
        return self._emb

    def forward(self, input_index, tag_index):
        # 此处x是词向量的索引
        # 取出来的形状就是SV
        vec = self._emb[input_index]
        tag = self._emb[tag_index]
        # print(vec.shape)
        # print(tag.shape)
        # 输入全连接之前需要reshape成一个矢量
        vec = vec.reshape(-1, 12)
        vec = self._layer(vec)
        # print(vec.shape)
        return vec, tag


if __name__ == '__main__':
    net = CBowNet()
    inputs = torch.randint(10, (2, 4))
    tags = torch.randint(10, (2,))
    print(inputs.shape, tags.shape)
    print(net(inputs, tags))
