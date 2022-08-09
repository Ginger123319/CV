import torch
from torch import nn


# 残差层：输入和输出通道数目一致
# 池化层：第一层改变输出通道为输入通道的2倍（借用步长减小特征图尺寸）；第二层就是输入和输出通道一致
# BN：除输出层都可以添加这一层，约束数据到一定范围，凸显数据差异;使用时要将网络层的bias设为False，BN中带有bias
# 保持输入尺寸和输出尺寸一致使用padding

# 残差层
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


# 池化层
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


# 网络层
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.layer = nn.Sequential(
        #     # 全连接MNIST使用
        #     # nn.Linear(784, 600),
        #     nn.Linear(32 * 32 * 3, 600),
        #     nn.ReLU(),
        #     nn.Linear(600, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 150),
        #     nn.ReLU(),
        #     nn.Linear(150, 80),
        #     nn.ReLU(),
        #     nn.Linear(80, 40),
        #     nn.ReLU(),
        #     nn.Linear(40, 10),
        #     nn.Softmax(dim=1)
        # )
        # # 构建卷积神经网络CNN
        # self.layer = nn.Sequential(
        #     # 卷积CIFAR10使用
        #     nn.Conv2d(3, 16, 3, 1, padding=1),
        #     # nn.Conv2d(1, 16, 3, 1),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, 1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 3, 1)
        # )
        # self.out_layer = nn.Sequential(
        #     # nn.Linear(256 * 18 * 18, 10),
        #     # 卷积CIFAR10使用
        #     nn.Linear(256 * 2 * 2, 10),
        #     nn.Softmax(dim=1)
        # )

        # 构建带有残差层和池化层以及BN的卷积神经网络CNN
        self.layer = nn.Sequential(
            # 卷积CIFAR10使用
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            # nn.Conv2d(1, 16, 3, 1),
            # nn.MaxPool2d(2),
            # nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
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
        )
        self.out_layer = nn.Sequential(
            # nn.Linear(256 * 18 * 18, 10),
            # 卷积CIFAR10使用
            nn.Linear(256 * 2 * 2, 10),
            # nn.Softmax(dim=1)
        )

    # 返回N*10的矩阵
    def forward(self, x):
        inner_out = self.layer(x)
        outer = self.out_layer(inner_out.reshape(inner_out.shape[0], -1))
        # return inner_out
        return outer


if __name__ == '__main__':
    net = Net()
    # print(net)
    test_data = torch.randn(3, 3, 32, 32)
    out = net.forward(test_data)
    print(out.shape)
    print(out)
