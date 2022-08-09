import torch
from torch import nn
import thop
from thop import clever_format


# 构建隐藏层
class Block(nn.Module):
    def __init__(self, c_in, i, t, c, n, s):
        super(Block, self).__init__()
        # 第几次调用
        self.i = i
        self.n = n
        # 是否为最后一次调用的标签
        self.flag = (self.i == self.n - 1)
        # 处理步长
        _s = s if self.flag else 1
        # print(_s)
        # 处理隐藏层的输入和输出通道数
        # 输入就是c_in，注意需要更新
        # 定义隐藏层中第一层输出通道数目
        _c_in = c_in * t
        # 最后一层输出处理
        _c = c if self.flag else c_in
        # 定义隐藏层的网络结构
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, _c_in, 1, 1, bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            # 注意此处为_s,不是s
            nn.Conv2d(_c_in, _c_in, 3, _s, padding=1, groups=_c_in, bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self, x):
        if self.flag is True:
            return self.layer(x)
        else:
            return self.layer(x) + x


# 定义参数列表t c n s
config = [[1, 16, 1, 1],

          [6, 24, 2, 2],

          [6, 32, 3, 2],
          [6, 64, 4, 2],
          [6, 96, 3, 1],
          [6, 160, 3, 2],
          [6, 320, 1, 1]]


# 构建整个网络


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        # 隐藏层
        # 用一个block来接收所有的隐藏层
        self.block = []
        # 隐藏层的入口通道数初始化
        c_in = 32
        # 根据参数列表实例化隐藏层
        # 切片的时候就是遍历一次列表元素
        # 套一层for循环就是在每遍历一个元素时，进行for循环，取出列表元素的每个值
        for t, c, n, s in config[0:]:
            for i in range(n):
                self.block.append(Block(c_in, i, t, c, n, s))
            # 更新输入通道
            c_in = c
        # 将隐藏层对象添加到网络层中
        # *表示取出列表中的每个元素
        self.block_layer = nn.Sequential(*self.block)
        # 输出层
        self.out_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            # 输出尺寸调整为1*1
            nn.AdaptiveAvgPool2d((1, 1)),
            # 最后输出就不加BatchNorm2d
            nn.Conv2d(1280, 10, 1, 1),
            # 输出函数
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        # return self.block_layer(self.input_layer(x))
        return self.out_layer(self.block_layer(self.input_layer(x)))


if __name__ == '__main__':
    # 测试
    test = torch.randn(1, 3, 224, 224)
    net = MobileNetV2()
    print(net)
    print(net.forward(test).shape)
    # 计算模型的浮点运算量以及参数量统计，默认单位为字节
    # clever_format换算为M
    print(clever_format(thop.profile(net, (test,))))
