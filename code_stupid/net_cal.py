import torch
from torch import nn
from thop import profile


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = nn.Conv2d(3, 16, 2, 2, 1, bias=False)

    def forward(self, x):
        return self._net(x)


if __name__ == '__main__':
    x = torch.randn(5, 3, 16, 16)
    net = Net()
    print(net(x).shape)
    flop, param = profile(net, inputs=(x,))
    print(flop, param)
