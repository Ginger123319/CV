from thop import profile, clever_format
import torch
from torch import nn

# 基础卷积块，Conv2d+BN+LeakyRelu
from torch.nn.functional import interpolate


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 残差块
# 用BN时，bias需要置为False
class ConvolutionalResLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.layer(x) + x


# Conv2dSet块
# 1*1的卷积核不需要加padding，因为不做像素融合，只做通道变化
class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.layer(x)


# 上采样块
class UpLayer(nn.Module):
    def __init__(self):
        super(UpLayer, self).__init__()
        self.layer = nn.Sequential()

    def forward(self, x):
        return interpolate(self.layer(x), scale_factor=2, mode="nearest")


# 主网络组装
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.layer52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),
            ConvolutionalResLayer(64),
            ConvolutionalLayer(64, 128, 3, 2, 1),
            ConvolutionalResLayer(128),
            ConvolutionalResLayer(128),
            ConvolutionalLayer(128, 256, 3, 2, 1),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256),
            ConvolutionalResLayer(256)
        )
        self.layer26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 2, 1),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512),
            ConvolutionalResLayer(512)
        )
        self.layer13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 2, 1),
            ConvolutionalResLayer(1024),
            ConvolutionalResLayer(1024),
            ConvolutionalResLayer(1024),
            ConvolutionalResLayer(1024)
        )
        # 涉及到分支的网络需要单独写一层网络
        self.set_layer13 = ConvolutionalSetLayer(1024, 512)
        self.detect_layer13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 45, 1, 1, 0)
        )
        # 写了一个UP网络层，这样写比调用函数插值方便
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpLayer()
        )
        self.set_layer26 = ConvolutionalSetLayer(768, 384)
        self.detect_layer26 = nn.Sequential(
            ConvolutionalLayer(384, 768, 3, 1, 1),
            nn.Conv2d(768, 45, 1, 1, 0)
        )
        self.up_52 = nn.Sequential(
            ConvolutionalLayer(384, 192, 1, 1, 0),
            UpLayer()
        )
        self.set_layer52 = ConvolutionalSetLayer(448, 224)
        self.detect_layer52 = nn.Sequential(
            ConvolutionalLayer(224, 448, 3, 1, 1),
            nn.Conv2d(448, 45, 1, 1, 0)
        )

    def forward(self, x):
        out_52 = self.layer52(x)
        out_26 = self.layer26(out_52)
        out_13 = self.layer13(out_26)
        out_set_13 = self.set_layer13(out_13)
        out_detect_13 = self.detect_layer13(out_set_13)

        out_up_26 = self.up_26(out_set_13)
        route_26 = torch.cat((out_up_26, out_26), dim=1)
        out_set_26 = self.set_layer26(route_26)
        out_detect_26 = self.detect_layer26(out_set_26)

        out_up_52 = self.up_52(out_set_26)
        route_52 = torch.cat((out_up_52, out_52), dim=1)
        out_set_52 = self.set_layer52(route_52)
        out_detect_52 = self.detect_layer52(out_set_52)
        return out_detect_13, out_detect_26, out_detect_52


# 测试
if __name__ == '__main__':
    data = torch.randn(2, 3, 416, 416)
    net = DarkNet53()
    print(net(data)[2].shape)
    flops, params = clever_format(profile(net, (data,)))
    print(flops, params)
