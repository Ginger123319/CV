import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, c_in, c_out, flag=False):
        super(DownSample, self).__init__()
        if flag is False:
            self.ds_layer = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, 2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.ds_layer = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, 1, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        return self.ds_layer(x)


class Outer(nn.Module):
    def __init__(self):
        super(Outer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, output_mode=False, stride=1):
        super(BasicBlock, self).__init__()
        self.flag = output_mode
        # 输入的通道数目和输出的通道数目一致的情况
        self.ds_net = DownSample(c_in, c_out, self.flag)
        if self.flag is True:
            self.layer = nn.Sequential(
                nn.Conv2d(c_in, c_in, 1, 1, bias=False),
                # 加了BN，上一层的bias需要设置为False，因为BN中已经带有了bias
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_in, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(c_in),
                # 输入输出通道数目一致
                # nn.Conv2d(c_out, c_in, 1, 1, bias=False),
                # nn.BatchNorm2d(c_in),
                # 输入输出通道数目不一致
                nn.Conv2d(c_in, c_out, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, 1, bias=False),
                # 加了BN，上一层的bias需要设置为False，因为BN中已经带有了bias
                nn.BatchNorm2d(c_out),
                nn.Conv2d(c_out, c_out, 3, 2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                # 输入输出通道数目一致
                # nn.Conv2d(c_out, c_in, 1, 1, bias=False),
                # nn.BatchNorm2d(c_in),
                # 输入输出通道数目不一致
                nn.Conv2d(c_out, c_in, 1, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )

    def forward(self, x):
        return self.layer(x)
        # + self.ds_net(x)


class Resnet50(nn.Module):
    def __init__(self, c_in):
        super(Resnet50, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            BasicBlock(64, 256, False),
            BasicBlock(256, 64, False),
            BasicBlock(64, 256, False),
            DownSample(256, 128, False),
            BasicBlock(128, 512, True),
            BasicBlock(512, 128, False),
            BasicBlock(128, 512, False)
        )

    def forward(self, x):
        # 将卷积结果改变形状后传入到全连接中
        # return Outer.forward(self.layer(x).reshape(-1, 2048))
        return self.layer(x)


if __name__ == '__main__':
    test_input = torch.randn(3, 64, 32, 32)
    # net = Resnet50(64)
    # # print(net.ds_net(test_input).shape)
    # ds_net = DownSample(64, 256)
    # # print(net)
    # print(net.forward(test_input).shape)  # torch.Size([3, 128, 32, 32])
    # # 进去和出来的形状一模一样，可以直接相加

    net = BasicBlock(64, 256, False, 1)
    print(net.forward(test_input).shape)
