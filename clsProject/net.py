import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 1, 1, 0),
            nn.Conv2d(4, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1, 1, 0),
            nn.Conv2d(8, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(16 * 40 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.reshape(-1, 16 * 40 * 2)
        out = self.out_layer(out)
        return out


if __name__ == '__main__':
    data = torch.randn(30, 1, 632, 22)
    net = Net()
    print(net(data).shape)
