#网络搭建--R网路

import torch
import torch.nn as nn
import torch.nn.functional as F


# R网路
class RNet(nn.Module):
    def __init__(self):
        super(RNet,self).__init__()
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1,padding=1),   # conv1
            nn.PReLU(),                                                                    # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),                                         # pool1
        )
        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),                                 # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),     # pool2
        )
        self.pre_layer3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()                                  # prelu3
        )
        self.conv4 = nn.Linear(64*3*3,128)  # conv4
        self.prelu4 = nn.PReLU()            # prelu4
        #detetion
        self.conv5_1 = nn.Linear(128,1)   # 置信度
        #bounding box regression
        self.conv5_2 = nn.Linear(128, 4)  # 偏移量

    def forward(self, x):
        #backend
        x = self.pre_layer1(x)
        # print("pre_layer1",x.shape)
        x = self.pre_layer2(x)
        # print("pre_layer2", x.shape)
        x = self.pre_layer3(x)
        # print("pre_layer3", x.shape)
        x = x.view(x.size(0),-1)
        x = self.conv4(x)
        x = self.prelu4(x)
        #detection
        label = F.sigmoid(self.conv5_1(x)) # 置信度
        offset = self.conv5_2(x)          # 偏移量

        return label,offset

# 测试
if __name__ == '__main__':
    x = torch.rand(1,3,24,24)
    net = RNet()
    cond, offset =net(x)
    print(cond.shape)
    print(offset.shape)