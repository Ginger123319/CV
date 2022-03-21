import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import optim

# x = np.arange(-9.9, 10, 0.1)
# # e的-x次方
# y = 1 / (1 + np.exp(-x))
# y1 = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# # sigmoid函数
# plt.plot(x, y)
# # tanh函数，双曲正切曲线
# plt.plot(x, y1)
# plt.show()

# _x = [i / 100 for i in range(100)]
# print(_x)
# _y = [3 * e + 4 + random.random() for e in _x]
# print(_y)
# # # 默认是画线，我们需要散点，所以增加了一个参数值
# # plt.plot(_x, _y, ".")
# # plt.show()
#
# # 需要求解w和b，实际上就是3和4+random
# # 假设w、b未训练前为一个随机数
# w = random.random()
# b = random.random()
#
# plt.ion()
# # zip函数的作用是组合两个列表,各自取出一个元素打包成为一个元组
# for i in range(30):
#     for x, y in zip(_x, _y):
#         # 前向运算
#         z = w * x + b
#         # 设计损失函数
#         o = z - y
#         # 目的是通过调账w与b的值，达到loss值相对较小，此时可以说该模型训练好了
#         loss = o ** 2
#
#         # 梯度下降方法
#         # 通过loss函数对w以及b计算梯度【求各自的偏导】
#         dw = 2 * o * x
#         db = 2 * o
#
#         # 此时w,b需要往梯度下降的方向去变化
#         w = -dw * 0.01 + w
#         b = -db * 0.01 + b
#
#         print(w, b, loss)
#         #
#         # plt.cla()
#         # plt.plot(_x, _y, ".")
#         # v = [w * e + b for e in _x]
#         # plt.plot(_x, v)
#         # plt.pause(0.01)
# plt.ioff()
#
# t = torch.tensor(3.0, requires_grad=True)
# print(t)
# f = t * 3 + 2
# # # 求导
# # f.backward()
# # # f对t求导的结果展示
# # print(t.grad)
#
# # 第二种方式
# print(torch.autograd.grad(f, t))

# 使用pytorch实现梯度下降算法
# 与random()函数不同，此处指定的是size即生成数字的个数，并且是一个张量
# 正常情况下会生产对应size的浮点数，大小在0到1之间，保留了四位小数
# print(torch.rand(99))

# # 张量计算后还是一个张量
# # 构建一堆散点
# xs = torch.arange(0.01, 1, 0.01)
# ys = 3 * xs + 4 + torch.rand(99)

# print(ys)
# plt.plot(xs, ys, ".")
# plt.show()

# class Line(torch.nn.Module):
#     def __init__(self):
#         super(Line, self).__init__()
#         # 前提条件是已知这是一个线性模型问题，给定一个初始的w以及b
#         self.w = torch.nn.Parameter(torch.rand(1))
#         self.b = torch.nn.Parameter(torch.rand(1))
#
#     # 前向计算，给一个输入得到一个输出
#     def forward(self, x):
#         return self.w * x + self.b
#
#
# if __name__ == '__main__':
#     # 创建模型
#     line = Line()
#     # 创建优化器，设计损失函数，降低损失的作用
#     # momentum=0.9设置学习的动量为0.9，加快学习速度
#     # opt = optim.SGD(line.parameters(), lr=0.1, momentum=0.9)
#     # Adam优化器会根据损失的值进行动态调整步长即lr【决定梯度下降的快慢】
#     opt = optim.Adam(line.parameters(), lr=0.1)
#
#     # loss_func = torch.nn.MSELoss()
#     plt.ion()  # 打开画图
#     for epoch in range(30):
#         for _x, _y in zip(xs, ys):
#             # 前向计算的结果
#             z = line.forward(_x)
#
#             loss = (z - _y) ** 2
#             # loss = loss_func(z, -y)
#             # 在每次更新梯度之前将上一个梯度清空
#             opt.zero_grad()
#             # 求梯度
#             loss.backward()
#             # 更新梯度
#             opt.step()
#             # 从张量中取出值，调用item()函数
#             print(line.w.item(), line.b.item(), loss.item())
#
#             plt.cla()
#             plt.plot(xs, ys, ".")
#             # 当前w和b前向计算后的结果集
#             v = [line.w.detach() * e + line.b.detach() for e in xs]
#             plt.plot(xs, v)
#             plt.pause(0.01)
#         plt.ioff()
#         # plt.show()
"""
梯度下降法：核心就是求解w，最终达到在该w的情况下，损失值降到很小
步骤：
（构造散点，仅在该案例中需要用到）
（1）选取模型，此处为线性模型
（2）给出w和b的一个初始值，根据模型和输入值【x】，前向计算算出该模型输出的值【z】
（3）根据输出值【z】与期望的值【y】的差值来设计损失函数
（4）计算损失函数的梯度（对w，b求偏导）
（5）根据梯度来调整参数【w，b】的大小
（6）然后产生新的w和b
（7）重复第二步到第六步，直到产生的损失降到一个理想的大小
（8）实时画点plt.plot
"""
# 使用pytorch实现梯度下降法
# 构造输入数据【每次输入两个x故应该使用两个w作为参数来设计模型】和标签
train_X = torch.tensor([[1., 1], [1, 0], [0, 1], [0, 0]])
train_Y = torch.tensor([[0.], [1], [1], [0]])
# 构建测试集
test_X = torch.tensor([[2, 1], [0, 0.5], [0.1, 0.1]])


# 神经网络类
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 此处需要两个参数，所以初始化两个w组成一个矢量
        self.W = torch.nn.Parameter(torch.randn(2))
        # 初始化一个偏移量b，也是一个矢量
        self.b = torch.nn.Parameter(torch.randn(1))

    # 前向计算，代入输入值计算经过模型运算的输出值（此处有问题）
    def forward(self, x):
        return (x * self.W).sum() + self.b


# 训练类
class Trainer:
    # 生成一个网络对象以及优化器SGD，lr为步长（学习速度）
    def __init__(self):
        self.net = Net()
        self.opt = optim.SGD(self.net.parameters(), lr=0.01)

    def train(self):

        for _ in range(10000):
            # 需要将输入值也就是训练集随机输入
            # i = torch.randint(0, 4, (1,))
            # 其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b
            i = random.randint(0, 3)
            _x = train_X[i]
            _y = train_Y[i]
            # 计算训练后的输出值以及损失
            y = self.net.forward(_x)
            loss = (y - _y) ** 2

            # 清除之前保存的梯度值
            self.opt.zero_grad()
            # 对损失函数求导，也就是求梯度
            loss.backward()
            # 根据梯度来修改w以及b的大小来将损失值减小
            self.opt.step()

        # 测试训练结果
        for x in train_X:
            print(self.net.forward(x))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

# # 普通实现,前期输入数据处理到0到1之间
# _x = [i / 100 for i in range(100)]
# # print(_x)
# _y = [4 * e + 3 + random.random() for e in _x]
# # print(type(_y))
# # plt.plot(_x, _y, ".")
# # plt.show()
#
# w = random.random()
# b = random.random()
# # 开始画图
# plt.ion()
# # 遍历所有的数据对
# for i in range(30):
#     for x, y in zip(_x, _y):
#         z = w * x + b
#         o = z - y
#         loss = o ** 2
#
#         dw = 2 * o * x
#         db = 2 * o
#
#         w = -dw * 0.1 + w
#         b = -db * 0.1 + b
#         print(w, b, loss)
#         # 每次画图之前需要清空画板
#         plt.cla()
#         # 画散点
#         plt.plot(_x, _y, ".")
#         # 画当前w，b情况下的直线
#         v = [w * e + b for e in _x]
#         plt.plot(_x, v)
#         # 每次画线之后需要暂停一段时间
#         plt.pause(0.01)
# # 关闭画图
# plt.ioff()
# # 展示循环中最后一次的展示的图像
# plt.show()
