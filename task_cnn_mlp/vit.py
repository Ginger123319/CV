# 将transformer用于图像方面
"""
两种方案，1直接在原图，分成n*n的方块，然后在每个块提取特征，然后在使用transformer
一般情况用的第一种办法，速度要比第二个更快
2,原图做卷积的到特征图，在特征图进行transformer
"""
import torch
from torch import nn
import cfg


class Block(nn.Module):
    def __init__(self, c_in, c_out, count):
        super(Block, self).__init__()
        self.s = 2 if c_in != c_out and count == 0 else 1
        self.c_in = c_in if count == 0 else c_out
        self.c_out = c_out
        self.block_layer = nn.Sequential(
            nn.Conv2d(self.c_in, self.c_out, 3, self.s, 1),
            nn.BatchNorm2d(self.c_out),
            nn.Hardswish(),
            nn.Conv2d(self.c_out, self.c_out, 3, 1, 1),
            nn.BatchNorm2d(self.c_out),
            nn.Hardswish(),
        )

    def forward(self, x):
        out = self.block_layer(x)
        if self.s == 2:
            return out
        else:
            return out + x


class VitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.MaxPool2d(3, 2, 1)

        )
        self.vit_layer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, activation="gelu")
            # nn.TransformerEncoder()
        )

    def vit(self, feature_map):
        # 在特征图处理
        # print(feature_map.shape)
        n, c, h, w = feature_map.shape
        feature = feature_map.reshape(n, c, h * w).permute(2, 0, 1)
        # 此时feature新装变成了 s nv,但是由于我的电脑版本没有，batch_firsr这个参数。智能输入s,n,v
        feature = self.vit_layer(feature)
        # 出来的形状也是 s,n,v将形状改变回来
        feature = feature.permute(1, 2, 0).reshape(n, c, h, w)
        return feature

    # 原图上处理
    def vit2(self, x):
        # 原图形状是  N C H W 对原图切分后 变成了 N C  Ah  bw
        # 一共六个维度
        # 先将，a, b,一道N旁边合并 a*b*N
        # 然后做transformer,在还原
        # print("X",x.shape)
        _N, _C, _H, _W = x.shape
        _h, _w = _H // 4, _W // 4
        x = x.reshape(_N, _C, 4, _h, 4, _w)
        # print("x",x.shape)
        # 原本需要的是 N S V ，但是由于版本问题，只能用 S ,N,V
        x = x.permute(0, 2, 4, 1, 3, 5)
        # 现在形状是n,a,b,c,h,w
        x = x.reshape(-1, _C, _h, _w)
        # print(x.shape)
        # 在做transformer之前要对数据做一次特征提取
        x = self.first_cnn_layer(x)
        # 进入特征提取网络，尺寸和通道都可能 发生变化
        # print(x.shape)
        n, c, h, w = x.shape
        x = self.vit(x)
        x = x.reshape(_N, 4, 4, c, h, w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(_N, c, 4 * h, 4 * w)

        return x

    def forward(self, x):
        # print(x.shape)
        # out1=self.vit(x)
        out1 = self.vit2(x)
        # print("out1", out1.shape)
        return out1


if __name__ == '__main__':
    # 图像形状是N C H W
    a = torch.randn(3, 3, 100, 100)
    net = VitNet()
    net(a)
    print("new!")
