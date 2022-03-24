import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            # nn.Linear(784, 600),
            nn.Linear(32 * 32 * 3, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.Softmax(dim=1)
        )
        # 构建卷积神经网络CNN
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            # nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1)
        )
        self.out_layer = nn.Sequential(
            # nn.Linear(256 * 18 * 18, 10),
            nn.Linear(256 * 22 * 22, 10),
            nn.Softmax(dim=1)
        )

    # 返回N*10的矩阵
    def forward(self, x):
        inner_out = self.layer(x)
        outer = self.out_layer(inner_out.reshape(inner_out.shape[0], -1))
        return outer


if __name__ == '__main__':
    net = Net()
    test_data = torch.randn(3, 1, 28, 28)
    out = net.forward(test_data)
    print(out.shape)
    print(out)
