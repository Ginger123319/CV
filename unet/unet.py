# import torch
# from torch import nn
# from torch.nn.functional import interpolate
#
#
# class Block(nn.Module):
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(c_in, c_out, 3, 1, 1, padding_mode="reflect"),
#             nn.BatchNorm2d(c_out),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(c_out, c_out, 3, 1, 1, padding_mode="reflect"),
#             nn.BatchNorm2d(c_out),
#             nn.LeakyReLU(0.1)
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# # 仅改变特征图尺寸
# class DownSample(nn.Module):
#     def __init__(self, c):
#         super(DownSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(c, c, 3, 2, 1, padding_mode="reflect"),
#             nn.BatchNorm2d(c),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
#
# class UpSample(nn.Module):
#     def __init__(self, c):
#         super(UpSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(c, c // 2, 3, 1, 1, padding_mode="reflect"),
#             nn.BatchNorm2d(c // 2),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x, r):
#         # 前插法输入尺寸
#         up = interpolate(x, scale_factor=2, mode="nearest")
#         x = self.layer(up)
#         return torch.cat((x, r), dim=1)
#
#
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.B1 = Block(3, 64)
#         # self.D1 = DownSample(64)
#         # 尺寸变为原来的一半，通道不变
#         self.D1 = nn.MaxPool2d(3, 2, 1)
#
#         self.B2 = Block(64, 128)
#         # self.D2 = DownSample(128)
#         self.D2 = nn.MaxPool2d(3, 2, 1)
#
#         self.B3 = Block(128, 256)
#         # self.D3 = DownSample(256)
#         self.D3 = nn.MaxPool2d(3, 2, 1)
#
#         self.B4 = Block(256, 512)
#         # self.D4 = DownSample(512)
#         self.D4 = nn.MaxPool2d(3, 2, 1)
#
#         self.B5 = Block(512, 1024)
#
#         self.U1 = UpSample(1024)
#         self.B6 = Block(1024, 512)
#
#         self.U2 = UpSample(512)
#         self.B7 = Block(512, 256)
#
#         self.U3 = UpSample(256)
#         self.B8 = Block(256, 128)
#
#         self.U4 = UpSample(128)
#         self.B9 = Block(128, 64)
#         # 输出层，不用BN和激活函数；N CHW
#         self.out_layer = nn.Conv2d(64, 3, 3, 1, 1)
#
#     def forward(self, x):
#         r1 = self.B1(x)
#         r2 = self.B2(self.D1(r1))
#         r3 = self.B3(self.D2(r2))
#         r4 = self.B4(self.D3(r3))
#         y1 = self.B5(self.D4(r4))
#
#         o1 = self.B6(self.U1(y1, r4))
#         o2 = self.B7(self.U2(o1, r3))
#         o3 = self.B8(self.U3(o2, r2))
#         o4 = self.B9(self.U4(o3, r1))
#         return self.out_layer(o4)
#
#
# if __name__ == '__main__':
#     test = torch.randn(1, 3, 256, 256)
#     net = UNet()
#     print(net.forward(test).shape)

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(3, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 128)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(128, 256)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(512, 1024)

        # right
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output


if __name__ == "__main__":
    a = torch.randn(1, 3, 256, 256)
    model = UNet()
    b = model(a)
    print(b.size())  # torch.randn(1, 3, 256, 256)
