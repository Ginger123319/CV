import torch
from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = models.resnet18()
        self.layer.fc = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid())

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    test = torch.randn(1, 3, 600, 600)
    net = Net()
    print(net(test))
