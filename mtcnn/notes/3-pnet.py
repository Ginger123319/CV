#网络搭建--P网路

import torch
import torch.nn as nn
import torch.nn.functional as F


# P网路
class PNet(nn.Module):
    def __init__(self):
        super(PNet,self).__init__()

        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,stride=1,padding=1), # conv1
            nn.PReLU(),                                                               # prelu1
            nn.MaxPool2d(kernel_size=3,stride=2),                                    # pool1
        )
        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),                                 # prelu2
        )
        self.pre_layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()                                   # prelu3
        )
        self.conv4_1 = nn.Conv2d(32,1,kernel_size=1)
        self.comv4_2 = nn.Conv2d(32,4,kernel_size=1)

    def forward(self, x):
        x = self.pre_layer1(x)
        # print("pre_layer1",x.shape)
        x = self.pre_layer2(x)
        # print("pre_layer2", x.shape)
        x = self.pre_layer3(x)
        # print("pre_layer3", x.shape)
        cond = F.sigmoid(self.conv4_1(x))     # 置信度，用sigmoid激活
        offset = self.comv4_2(x)             # 偏移量，不需要激活
        return cond,offset


# 测试
if __name__ == '__main__':
    x = torch.rand(1,3,12,12)
    net = PNet()
    cond, offset =net(x)
    print(cond.shape)
    print(offset.shape)