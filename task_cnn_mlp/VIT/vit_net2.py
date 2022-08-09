import torch
from torch import nn
from VIT.vit_transformer import Net


class VitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._ts_layer = Net(64)

    # 返回N*10的矩阵
    def forward(self, x):
        _n, _c, _H, _W = x.shape
        _h, _w = _H // 4, _W // 4
        # 切分原图
        feature = x.reshape(_n, _c, 4, _h, 4, _w)
        feature = feature.permute(0, 2, 4, 1, 3, 5)
        # 将多出来的通道合并到N所在的通道上
        feature = feature.reshape(-1, _c, _h, _w)
        # 放入vit网络进行全局相关性处理，qkv都是自身，自注意力
        feature = self._ts_layer(feature)
        # 还原为原来的批次
        _, c, h, w = feature.shape
        feature = feature.reshape(_n, 4, 4, c, h, w)
        feature = feature.permute(0, 3, 1, 4, 2, 5)
        feature = feature.reshape(_n, c, 4 * h, 4 * w)
        # 进行后续的卷积操作 
        return feature


if __name__ == '__main__':
    net = VitNet()
    test_data = torch.randn(3, 64, 100, 100)
    out = net.forward(test_data)
    print(out.shape)
    # print(out)
