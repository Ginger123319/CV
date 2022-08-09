import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_layer = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.reshape(-1, 128*1*1)
        out = self.out_layer(out)
        return out


if __name__ == '__main__':
    data = torch.randn(30, 1, 632, 22)
    net = Net()
    print(net(data).shape)
