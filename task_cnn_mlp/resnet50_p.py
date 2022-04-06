import torch
from torch import nn
from torchvision import models

config = [[64, 256, 1, 3], [128, 512, 2, 4], [256, 1024, 2, 6], [512, 2048, 2, 3]]


# 实现与resnet50相同结构的网络模型
class BasicBlock(nn.Module):
    def __init__(self, c_in, middle_out, c_out, s, n, i):
        super(BasicBlock, self).__init__()
        self.i = i
        self.n = n
        self.flag = (self.i == 0)
        s = s if self.flag else 1
        c_in = c_in if self.flag else c_out
        # 做除法结果都为浮点数，需要强转为int才能作为通道数
        # 将计算最复杂的部分直接作为变量传进来简化代码量，比如中间输出通道数目
        # 实现不了的直接传入一个变量即可
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, middle_out, 1, 1, bias=False),
            nn.BatchNorm2d(middle_out),
            nn.Conv2d(middle_out, middle_out, 3, s, 1, bias=False),
            nn.BatchNorm2d(middle_out),
            nn.Conv2d(middle_out, c_out, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        # 第一次调用block有的是通道变化，有的是通道和尺寸一起变化
        if self.flag:
            return self.layer(x)
        else:
            return self.layer(x) + x


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        c_in = 64
        self.block = []
        for m, c_out, s, n in config[0:]:
            for i in range(n):
                self.block.append(BasicBlock(c_in, m, c_out, s, n, i))
            c_in = c_out
        self.hidden_layer = nn.Sequential(*self.block, nn.AdaptiveAvgPool2d((1, 1)))
        # print(self.hidden_layer)
        self.output_layer = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out.reshape(-1, 2048 * 1 * 1))
        return out


if __name__ == '__main__':
    # ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
    # data = torch.randn(1, 3, 32, 32)
    data = torch.randn(1, 3, 56, 56)
    net = BasicBlock(256, 128, 512, 1, 3, 1)
    net = Resnet50()
    print(net)
    # print(net.forward(data).shape)
    # print(models.resnet50())
