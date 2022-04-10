# coding=gbk
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# 定义卷积模块
class Block(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, rate=1):
        super(Block, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * rate, dilation=1 * rate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        out = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return out


# 定义上采用层
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


# RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.Blockin = Block(in_ch, out_ch, rate=1)

        self.Block1 = Block(out_ch, mid_ch, rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block2 = Block(mid_ch, mid_ch, rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block3 = Block(mid_ch, mid_ch, rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block4 = Block(mid_ch, mid_ch, rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block5 = Block(mid_ch, mid_ch, rate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block6 = Block(mid_ch, mid_ch, rate=1)

        self.Block7 = Block(mid_ch, mid_ch, rate=2)

        self.Block6d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block5d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block4d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block3d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block2d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block1d = Block(mid_ch * 2, out_ch, rate=1)

    def forward(self, x):
        hx = x
        hxin = self.Blockin(hx)

        hx1 = self.Block1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.Block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.Block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.Block4(hx)
        hx = self.pool4(hx4)

        hx5 = self.Block5(hx)
        hx = self.pool5(hx5)

        hx6 = self.Block6(hx)

        hx7 = self.Block7(hx6)

        hx6d = self.Block6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.Block5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.Block4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.Block3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.Block2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.Block1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.Blockin = Block(in_ch, out_ch, rate=1)

        self.Block1 = Block(out_ch, mid_ch, rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block2 = Block(mid_ch, mid_ch, rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block3 = Block(mid_ch, mid_ch, rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block4 = Block(mid_ch, mid_ch, rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block5 = Block(mid_ch, mid_ch, rate=1)

        self.Block6 = Block(mid_ch, mid_ch, rate=2)

        self.Block5d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block4d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block3d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block2d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block1d = Block(mid_ch * 2, out_ch, rate=1)

    def forward(self, x):
        hx = x

        hxin = self.Blockin(hx)

        hx1 = self.Block1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.Block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.Block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.Block4(hx)
        hx = self.pool4(hx4)

        hx5 = self.Block5(hx)

        hx6 = self.Block6(hx5)

        hx5d = self.Block5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.Block4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.Block3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.Block2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.Block1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.Blockin = Block(in_ch, out_ch, rate=1)

        self.Block1 = Block(out_ch, mid_ch, rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block2 = Block(mid_ch, mid_ch, rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block3 = Block(mid_ch, mid_ch, rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block4 = Block(mid_ch, mid_ch, rate=1)

        self.Block5 = Block(mid_ch, mid_ch, rate=2)

        self.Block4d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block3d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block2d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block1d = Block(mid_ch * 2, out_ch, rate=1)

    def forward(self, x):
        hx = x

        hxin = self.Blockin(hx)

        hx1 = self.Block1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.Block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.Block3(hx)
        hx = self.pool3(hx3)

        hx4 = self.Block4(hx)

        hx5 = self.Block5(hx4)

        hx4d = self.Block4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.Block3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.Block2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.Block1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.Blockin = Block(in_ch, out_ch, rate=1)

        self.Block1 = Block(out_ch, mid_ch, rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block2 = Block(mid_ch, mid_ch, rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Block3 = Block(mid_ch, mid_ch, rate=1)

        self.Block4 = Block(mid_ch, mid_ch, rate=2)

        self.Block3d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block2d = Block(mid_ch * 2, mid_ch, rate=1)
        self.Block1d = Block(mid_ch * 2, out_ch, rate=1)

    def forward(self, x):
        hx = x

        hxin = self.Blockin(hx)

        hx1 = self.Block1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.Block2(hx)
        hx = self.pool2(hx2)

        hx3 = self.Block3(hx)

        hx4 = self.Block4(hx3)

        hx3d = self.Block3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.Block2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.Block1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.Blockin = Block(in_ch, out_ch, rate=1)

        self.Block1 = Block(out_ch, mid_ch, rate=1)
        self.Block2 = Block(mid_ch, mid_ch, rate=2)
        self.Block3 = Block(mid_ch, mid_ch, rate=4)

        self.Block4 = Block(mid_ch, mid_ch, rate=8)

        self.Block3d = Block(mid_ch * 2, mid_ch, rate=4)
        self.Block2d = Block(mid_ch * 2, mid_ch, rate=2)
        self.Block1d = Block(mid_ch * 2, out_ch, rate=1)

    def forward(self, x):
        hx = x

        hxin = self.Blockin(hx)

        hx1 = self.Block1(hxin)
        hx2 = self.Block2(hx1)
        hx3 = self.Block3(hx2)

        hx4 = self.Block4(hx3)

        hx3d = self.Block3d(torch.cat((hx4, hx3), 1))
        hx2d = self.Block2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.Block1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


# U^2-Net #
class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        # encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


# U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


if __name__ == '__main__':
    data = torch.randn(1, 3, 224, 224)
    net = U2NET()
    # print(len(net.forward(data)))
    print(net(data)[0].shape)
