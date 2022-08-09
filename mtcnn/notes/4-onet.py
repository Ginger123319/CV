# 网络搭建--O网路

import torch
import torch.nn as nn
import torch.nn.functional as F


# O网路
class ONet(nn.Module):
    def __init__(self):
        super(ONet,self).__init__()
        # backend
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1),  # conv1
            nn.PReLU(),                                          # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),               # pool1
        )
        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),                                 # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),     # pool2
        )
        self.pre_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),                                 # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),     # pool3
        )
        self.pre_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()                                   # prelu4
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)  # conv5
        self.prelu5 = nn.PReLU()                 # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer1(x)
        # print("pre_layer1", x.shape)
        x = self.pre_layer2(x)
        # print("pre_layer2", x.shape)
        x = self.pre_layer3(x)
        # print("pre_layer3", x.shape)
        x = self.pre_layer4(x)
        # print("pre_layer4", x.shape)

        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        label = F.sigmoid(self.conv6_1(x)) # 置信度
        offset = self.conv6_2(x)          # 偏移量
        return label, offset

# 测试
if __name__ == '__main__':
    x = torch.rand(1,3,48,48)
    net = ONet()
    cond, offset =net(x)
    print(cond.shape)
    print(offset.shape)