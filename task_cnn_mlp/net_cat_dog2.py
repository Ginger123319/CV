import torch
from torch import nn
# from VIT import vit_deal
# from VIT import vit_net
from vit import VitNet


# 残差层
# 池化层
# BN
class ResBlock(nn.Module):
    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, padding=1, bias=False),
            # nn.Dropout(0.2),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x) + x


class Pool(nn.Module):
    def __init__(self, c_in, c_out):
        super(Pool, self).__init__()
        self.layer = nn.Sequential(
            # 下采样操作
            nn.Conv2d(c_in, c_out, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 构建带有残差层和池化层以及BN的卷积神经网络CNN
        self.vit_layer = VitNet()
        self.layer = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            Pool(64, 128),

            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            Pool(128, 256),

            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            Pool(256, 512),
            ResBlock(512),
            ResBlock(512)
        )
        self.out_layer = nn.Sequential(
            # nn.Linear(256 * 18 * 18, 10),
            # 卷积CIFAR10使用
<<<<<<< HEAD
            nn.Linear(512* 3 * 3, 1),
=======
            nn.Linear(512 * 2 * 2, 1),
>>>>>>> 88f083cfad1253dcd577913ff5a22f6969351530
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    # 返回N*10的矩阵
    def forward(self, x):
        x = self.vit_layer(x)
        inner_out = self.layer(x)
        outer = self.out_layer(inner_out.reshape(inner_out.shape[0], -1))
        # return inner_out
        return outer


if __name__ == '__main__':
    net = Net()
    # print(net)
    test_data = torch.randn(5, 3, 100, 100)
    out = net.forward(test_data)
    print(out.shape)
    # print(out)
