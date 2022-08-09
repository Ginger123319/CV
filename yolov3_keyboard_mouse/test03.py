import torch
from thop import profile
from torch import nn

x = torch.tensor([[[[1, 2], [3, 4]]]]).float()
# print(x)
x = torch.nn.functional.interpolate(x, scale_factor=2)


# print(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 6, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.randn(1, 3, 4, 4)
    net = Net()
    print(net(x).shape)
    print(profile(net, inputs=(x,)))
