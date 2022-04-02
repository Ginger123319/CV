import torch
from torch import nn
from torch.nn.functional import interpolate


class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(c_out, c_out, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.layer(x)


# 仅改变特征图尺寸
class DownSample(nn.Module):
    def __init__(self, c):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(c),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, c):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(c // 2),
            nn.LeakyReLU()
        )

    def forward(self, x, r):
        # 前插法输入尺寸
        up = interpolate(x, scale_factor=2, mode="nearest")
        x = self.layer(up)
        return torch.cat((x, r), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.B1 = Block(3, 64)
        self.D1 = DownSample(64)

        self.B2 = Block(64, 128)
        self.D2 = DownSample(128)

        self.B3 = Block(128, 256)
        self.D3 = DownSample(256)

        self.B4 = Block(256, 512)
        self.D4 = DownSample(512)

        self.B5 = Block(512, 1024)

        self.U1 = UpSample(1024)
        self.B6 = Block(1024, 512)

        self.U2 = UpSample(512)
        self.B7 = Block(512, 256)

        self.U3 = UpSample(256)
        self.B8 = Block(256, 128)

        self.U4 = UpSample(128)
        self.B9 = Block(128, 64)
        # 输出层，不用BN和激活函数；N CHW
        self.out_layer = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        r1 = self.B1(x)
        r2 = self.B2(self.D1(r1))
        r3 = self.B3(self.D2(r2))
        r4 = self.B4(self.D3(r3))
        y1 = self.B5(self.D4(r4))

        o1 = self.B6(self.U1(y1, r4))
        o2 = self.B7(self.U2(o1, r3))
        o3 = self.B8(self.U3(o2, r2))
        o4 = self.B9(self.U4(o3, r1))
        return self.out_layer(o4)


if __name__ == '__main__':
    test = torch.randn(1, 3, 256, 256)
    net = UNet()
    print(net.forward(test).shape)
