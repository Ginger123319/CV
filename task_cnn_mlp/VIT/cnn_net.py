import torch
from torch import nn


class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 构建卷积神经网络CNN
        self.layer = nn.Sequential(
            # 卷积CIFAR10使用
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    # 返回N*10的矩阵
    def forward(self, x):
        outer = self.layer(x)
        return outer


if __name__ == '__main__':
    net = CnnNet()
    test_data = torch.randn(3, 3, 32, 32)
    out = net.forward(test_data)
    print(out.shape)
    # print(out)
