import torch
from torch import nn
from torchvision import models


class DownSample(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownSample, self).__init__()
        self.ds_layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 2, bias=False),
            nn.BatchNorm2d(c_out)

        )

    def forward(self, x):
        return self.ds_layer(x)


# 分析，残差块有两种，一种是带有下采样的（stride=2），一种不带，用标签来区分
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_down=False):
        super(BasicBlock, self).__init__()
        self.flag = is_down
        self.ds_net = DownSample(c_in, c_out)
        if self.flag is True:
            self.layer = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(c_out, c_out, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, 1, padding=1, bias=True),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )

    def forward(self, x):
        if self.flag is False:
            return self.layer(x) + x
        else:
            return self.layer(x) + self.ds_net(x)


class Outer(nn.Module):
    def __init__(self):
        super(Outer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)


class Resnet18(nn.Module):
    def __init__(self, c_in):
        super(Resnet18, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))

        )

    def forward(self, x):
        out = self.layer(x).reshape(-1, 512)
        return Outer().forward(out)


class Resnet34(nn.Module):
    def __init__(self, c_in):
        super(Resnet34, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))

        )

    def forward(self, x):
        out = self.layer(x).reshape(-1, 512)
        return Outer().forward(out)


if __name__ == '__main__':
    # net = BasicBlock(128, 256, True)
    # print(net)
    test_input = torch.randn(3, 128, 64, 64)
    # print(net.forward(test_input).shape)
    # # print(models.BasicBlock())
    net = Resnet34(128)
    # print(net)
    print(net.forward(test_input).shape)
