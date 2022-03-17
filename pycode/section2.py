import numpy as np
import random
import matplotlib.pyplot as plt
import torch

# x = np.arange(-9.9, 10, 0.1)
# # e的-x次方
# y = 1 / (1 + np.exp(-x))
# y1 = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# # sigmoid函数
# plt.plot(x, y)
# # tanh函数，双曲正切曲线
# plt.plot(x, y1)
# plt.show()

_x = [i / 100 for i in range(100)]
print(_x)
_y = [3 * e + 4 + random.random() for e in _x]
print(_y)
# # 默认是画线，我们需要散点，所以增加了一个参数值
# plt.plot(_x, _y, ".")
# plt.show()

# 需要求解w和b，实际上就是3和4+random
# 假设w、b未训练前为一个随机数
w = random.random()
b = random.random()

plt.ion()
# zip函数的作用是组合两个列表,各自取出一个元素打包成为一个元组
for i in range(30):
    for x, y in zip(_x, _y):
        # 套模型的到的输出
        z = w * x + b
        # 与y【目标输出值】的差值
        o = z - y
        # 设计损失函数,目的是通过调账w与b的值，达到loss值相对较小，此时可以说该模型训练好了
        loss = o ** 2

        # 梯度下降方法
        # 通过loss函数对w以及b计算梯度【求各自的偏导】
        dw = 2 * o * x
        db = 2 * o

        # 此时w,b需要往梯度下降的方向去变化
        w = -dw * 0.01 + w
        b = -db * 0.01 + b

        print(w, b, loss)
        #
        # plt.cla()
        # plt.plot(_x, _y, ".")
        # v = [w * e + b for e in _x]
        # plt.plot(_x, v)
        # plt.pause(0.01)
plt.ioff()

t = torch.tensor(3.0, requires_grad=True)
print(t)
f = t * 3 + 2
# # 求导
# f.backward()
# # f对t求导的结果展示
# print(t.grad)

# 第二种方式
print(torch.autograd.grad(f, t))
