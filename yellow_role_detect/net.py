import torch,thop
from torch import nn

class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(300*300*3,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 96),
            nn.ReLU(),
            nn.Linear(96, 78),
            nn.ReLU(),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 52),
            nn.ReLU(),
            nn.Linear(52,4)
        )

    def forward(self,x):
        return self.fc_layers(x)

class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(256*4*4,4),
            nn.Sigmoid()
        )
    def forward(self,x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.reshape(-1,256*4*4)
        out = self.out_layer(conv_out)
        return out


if __name__ == '__main__':
    # net = Net_v1()
    # x = torch.randn(3,300*300*3)
    # y = net(x)
    # print(y.shape)
    # flops, params = thop.profile(net, (x,))
    # print(flops)
    # print(params)
    net = Net_v2()
    x = torch.randn(1,3,300,300)
    y = net(x)
    print(y.shape)
    print(y)
    print(torch.sum(y))