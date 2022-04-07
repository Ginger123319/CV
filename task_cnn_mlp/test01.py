import torch
from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # net = Net()
    demo = torch.randn(1, 6, 3, 3)
    # print(net.forward(demo).shape)
    # print(models.resnet50())
    # demo = torch.randn(480)
    # print(demo.shape)
    # # 张量的条件筛选，大于0.5的值会转为True，再转为浮点就变成了1.
    # print((demo > 0.5).float())
    # print((demo[:, None]).shape)

    # 深度可分离卷积
    layer = nn.Conv2d(6, 6, 3, 1, groups=6)
    out = layer(demo)
    print(out.shape)
    print(out)
    # 使用reshape操作进行通道混洗
    out_v1 = out.reshape(1, 2, 3, 1, 1)
    # print(out_v1)
    out_v2 = out_v1.permute(0, 2, 1, 3, 4)
    # print(out_v2.shape)
    result = out_v2.reshape(1, 6, 1, 1)
    print(result)
    print(torch.Tensor([1]))
