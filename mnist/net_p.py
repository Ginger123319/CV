# 手写数字识别问题
# 28*28*1的图片分为0~9十个类别的问题
# 使用全连接神经网络构建模型
# 创建一个全连接神经网络类，继承于nn.Module类
from torch import nn


class MnistNet(nn.Module):
    def __init__(self):
        # 调用父类的初始化函数
        super(MnistNet, self).__init__()
        # 定义一个属性net来创建全连接网络层
        self.layer = nn.Sequential(
            # 由于输入是（N,784），因此输入层的输入为784
            nn.Linear(784, 392),
            # 每层创建后用激活函数激活
            nn.ReLU(),
            # 隐藏层
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, 98),
            nn.ReLU(),
            nn.Linear(98, 49),
            nn.ReLU(),
            # 输出层，十分类问题，最终输出为10
            nn.Linear(49, 10),
            # 输出后建议套一层输出函数，避免负数；第一轮训练后的损失会比不加要小
            # 需要对1轴上的数据进行归一化处理，需要一个轴参数
            nn.Softmax(dim=1)
        )

    # 前向计算
    def forward(self, x):
        # 向网络中传入输入，返回输出
        return self.layer(x)


if __name__ == '__main__':
    print(1)
