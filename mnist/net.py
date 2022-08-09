import torch
from torch import nn


# 定义创建神经网络的类
# 该类继承于torch.nn.Module
class MnistNet(nn.Module):
    # 初始化函数，需要使用super()【此处表示是一个对象，所以要有括号】将父类的初始化函数一块调用
    def __init__(self):
        super(MnistNet, self).__init__()
        # 定义神经网络的层数【layer】以及各层参数的个数
        # 调用nn.Sequential()实现对神经网络输入层、隐藏层以及输出层的定义
        self.layer = nn.Sequential(
            # 输入层
            # in_features=784, out_features=512, bias=True【bias指的就是偏移量b，为True代表函数中有偏移量】
            nn.Linear(28 * 28 * 1, 512),
            # 使用激活函数ReLU使得线性函数具有非线性能力，学习速度更快
            nn.ReLU(),
            # 隐藏层
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # 输出层，由于问题是一个“十分类问题”，输出结果元素个数应该是10个
            nn.Linear(32, 10),
            # 因为输出结果是由矩阵运算来的，所以可能出现负数
            # 判断结果是由概率大小比较得来的，概率只在0到1之间
            # 故需要将输出结果进行归一化处理，避免出现负数影响判断
            nn.Softmax(dim=1)
        )

    # 定义模型前向计算函数，得到经过模型学习后的输出值
    def forward(self, x):
        # 将输入值代入到模型中输出结果，并将结果返回
        out = self.layer(x)
        return out


if __name__ == '__main__':
    # 对该神经网络的类进行测试
    net = MnistNet()
    # 打印网络的结构
    print(net)

    # 定义一个随机的矩阵，符合输入数据的形状
    x = torch.randn(1, 784)
    out = net.forward(x)
    print(out.shape)
    # 前提条件:
    # 1轴的索引正好对应0~9，故就将对应索引的值作为该索引值的概率，即最有可能是0~9中的哪一类
    # 哪个索引的数字大的就看作该结果属于哪一类；也就是这一张图片属于哪一个数字类别
    print(out)
    print(torch.argmax(out, dim=1))
