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
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        pass


# 上采样块


class UpLayer(nn.Module):
    def __init__(self):
        super(UpLayer, self).__init__()

    def forward(self, x):
        return interpolate(x, scale_factor=2, mode="nearest")


# 主网络组装
# 测试
if __name__ == '__main__':
    data = torch.randn(2, 3, 416, 416)
