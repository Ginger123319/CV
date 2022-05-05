import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(22, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )
        self.layer = nn.Sequential(
            nn.Conv2d(22, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
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
        out1 = self.layer1(x)
        # thresh = torch.mean(out1)
        # out1 = (out1 > thresh).float()
        out2 = out1 * x
        # out = out2.permute(0, 2, 3, 1)
        out = self.layer(out2)
        out = out.reshape(-1, 128 * 1 * 1)
        out = self.out_layer(out)
        return out, out1, out2
        # return out.shape


if __name__ == '__main__':
    data = torch.randn(30, 22, 1, 632)
    net = Net()
    print(net(data)[2].shape)
