import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 构建卷积神经网络CNN
        self.layer = nn.Sequential(
            # 卷积CIFAR10使用
            nn.Conv2d(3, 32, 3, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_layer1 = nn.Sequential(
            # 卷积CIFAR10使用
            nn.Linear(256 * 4 * 4, 4),
            nn.Sigmoid()
        )
        self.out_layer2 = nn.Sequential(
            # 卷积CIFAR10使用
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    # 返回N*10的矩阵
    def forward(self, x):
        inner_out = self.layer(x).reshape(-1, 256 * 4 * 4)
        outer1 = self.out_layer1(inner_out)
        outer2 = self.out_layer2(inner_out)
        return outer1, outer2


if __name__ == '__main__':
    net = Net()
    test_data = torch.randn(2, 3, 300, 300)
    out = net.forward(test_data)
    print(out)
