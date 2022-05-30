import torch
from torch import nn

# 定义参数列表t c n s
config = [[-1, 32, 1, 2],
          [1, 16, 1, 1],
          [6, 24, 2, 2],
          [6, 32, 3, 2],
          [6, 64, 4, 2],
          [6, 96, 3, 1],
          [6, 160, 3, 2],
          [6, 320, 1, 1]]


# 由于两种类型的网络块只有步长不一致，故可以只设计一个网络块，传入一个步长变量
# 由于存在调用次数不同步长不一致的情况，需要定义一个i来存储该模块的调用次数
# 属于中间模块，输入通道需要定义一个变量传进来
class Block(nn.Module):
    def __init__(self, c_in, i, t, c, n, s):
        super().__init__()
        self.i = i
        self.n = n
        # 步长处理
        self._s = s if i == (n - 1) else 1
        # 构建网络时的通道处理
        # 通道变化也是在n-1次时发生的，之前的次数就是在进行通道升上去降下来，降下来的输出就等于输入的通道
        _c = c if i == (n - 1) else c_in
        # 1*1升高通道的输出通道数就为输入通道数的t倍
        _c_in = c_in * t
        # 开始构建网络块
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, _c_in, 1, 1, bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            # 少加了一个分组参数，需要注意此处为深度可分离卷积
            nn.Conv2d(_c_in, _c_in, 3, self._s, padding=1, groups=_c_in, bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)

        )

    def forward(self, x):
        if self.i != self.n - 1:
            # if self._s == 1:
            # 不能仅通过步长进行判断，因为存在步长为1，但是通道发生了变化的网络层
            # 因此就不能进行残差操作
            return self.layer(x) + x
        else:
            return self.layer(x)


class MobileNetV2(nn.Module):
    def __init__(self, net_config):
        super().__init__()
        self.config = net_config
        self.input_layer = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
        )
        self.block = []
        # 拿到输入层输出通道数
        c_in = self.config[0][1]
        # 取到输入层的输出通道即为下一层的输入通道
        for t, c, n, s in self.config[1:]:
            for i in range(n):
                self.block.append(Block(c_in, i, t, c, n, s))
            # n次网络块调用完毕后进行一次输入通道的更新，即下层网络的输入即为上一层的输出
            c_in = c
        self.hidden_layer = nn.Sequential(*self.block)
        self.out_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1280, 10, 1, 1, bias=False),
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        return self.out_layer(self.hidden_layer(self.input_layer(x)))


if __name__ == '__main__':
    test = torch.randn(1, 3, 224, 224)
    net = MobileNetV2(config)
    print(net)
    print(net.forward(test).shape)
