import random

import numpy as np
import torch
from torch import tensor

print(torch.__version__)
# print(torch.cuda.is_available())
# numpy既不能直接支持BP算法也不能支持GPU运算；工程上会使用pytorch进行操作

# 张量及相关操作
# 张量有形状，与python的列表含义不同
# 保证张量的元素数据类型一致，各个维度上元素的长度要一致
# a = [[1, 2, "s"], [1, 2]]
# print(a)
# # 创建张量时对元素个数和类型有限制
# # ndarrays with different lengths or shapes)
# # b = np.array([[1, 2, 3], [1, 2]])
# # b = np.array(9)--9
# b = np.array([[1, 2, 3], [1, 2, 3]])
# print(b, b.shape)

# c = np.zeros((2, 3))
# print(c)

# # 修改张量，通过指定轴上的索引值确定元素
# print(b[1][2])
# print(b[1, 2])
# b[1, 2] = 5
# print(b)
# # 张量切片，注意和列表切片区分开
# print(b[:, 1])
# print(b[1, :])

# 张量合并
# d = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]], [[3, 4]]])
# e = torch.tensor([[4, 5], [5, 6], [2, 3]])
# g = torch.tensor([[[4]], [[4]]])
# h = torch.tensor([[1, 2, 3]])
# # print(d.shape, e.shape)
# # 当有多个轴的时候，在某个轴上进行合并，参数就写哪个轴【合并轴】
# # 轴的数目要相同且非合并轴上的元素个数要一致
# # cat与np.concatenate【axis轴】功能一致，参数不同罢了
# f = torch.cat((d, e), dim=0)
#
# # 其他轴上的元素个数要一致，如果不一致，需要对矩阵进行变形
# i = torch.tensor([8, 9])
# print(i.shape)
# # 由于i只有一条轴，无法和e进行合并，需要增加一条轴
# # 逗号隔开的就是两个轴，:代表切片；此处的意思表示1轴上的所有索引就对应i所有的数字
# # 即1轴上索引0对应8，索引1对应9；故_i为[[8,9]];
# # 也可以理解为1轴上有两个元素，故1轴的形状为2；而总共只有两个元素两个轴，故只能是1x2的形状
# _i = i[None, :]
# # 而_i = i[: , None]表示0轴索引0对应[8]，索引1对应[9]；故此时_i为[[8], [9]]
# # _i = i[:, None, None]代表0轴索引0对应[[8]]
# # _i = i[None, :, None]代表1轴索引0对应[8]
# # _i = i[None, :, None]
# print(_i.shape, _i)
# print(torch.cat((d, _i), dim=0))
# print(torch.cat([g, h], dim=1))

# print(f)
# print(f.shape)

# # stack函数:同维度的元素按照对应的索引位置堆叠在一起
# # 堆叠后的结果会比原始张量升高一个维度；高维空间是由低维空间堆叠而来
g = torch.tensor([[[4]], [[1]]])
h = torch.tensor([[[1]], [[3]]])
print("g is:")
print(g)
print("h is:")
print(h)
# print(g[1])
# # 为了保持原始张量的维度
# print(g[[1]])
# # stack expects each tensor to be equal size, but got [2, 2, 2] at entry 0 and [2, 2] at entry 1
# # print(torch.stack((g, e), dim=0))
# # dim可以取[-4,3]取负数是什么含义-就和切片的-1，-2，-3一样，也是索引值
# print(g.shape)
# print(h.shape)
print("torch.stack((g, h), dim=1) result:")
print(torch.stack((g, h), dim=2))
print("torch.cat([g, h], dim=1) result:")
print(torch.cat([g, h], dim=1))

# # randn()函数可以生成指定形状的张量[至少是矢量]，其中元素值是随机生成的
# print(torch.randn(2))
# print(torch.randn(1))
# print(torch.tensor(1))
# print(torch.randn((2, 2)))
# print(np.random.randn(2, 2))

# randint()函数可以生成指定形状的张量[至少是矢量]，其中元素值是随机生成的
# print(torch.randint(0, 4, (1,)))
# 其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b
print(random.randint(0, 4))

# 张量的运算
# 张量【形状相同的时候】可以进行基本的加减乘除操作
# print(d+e)
# print(d-e)
# print(d*e)
# print(d/e)

# 形状不同时就要看是否能够张量运算是否符合广播的条件
# 张量广播的条件：各个张量对应的维度【0维：标量 1维：矢量 2维：矩阵 3维及以上就是n维张量】的元素个数要么一致要么为1
# 查看是否能广播时需要求出各个张量的形状，从右往左观察，形状中处在最右边数字的就是处于同一个维度的元素个数；
# 这些一般不在同一个轴上
# j = torch.tensor(5)
# print(j.shape)
# k = torch.arange(1, 9).reshape(2, 4)
# l = torch.arange(1, 3).reshape(2, 1)
# print(k.shape, k)
# print(l.shape, l)
# print(k + l)
#
# # # 当无法进行广播，但是想进行相加的，需要进行repeat操作【前提是对应维度上元素个数（size）存在倍数关系】
# m = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
# n = torch.tensor([[3, 2], [1, 2], [3, 2], [3, 4]])
# # repeat 在1轴上重复两次，每个元素复制一次[3,2]->[3,3,2,2]
# numpy中为np.tile(axis,repeat)函数，功能一致
# o = torch.repeat_interleave(n, 2, 1)
# print(o[0, 0])
# print(m + o)
#
# # 矩阵叉乘就是mxn@nxk=mxk
# print(m @ n)
# # 张量的叉乘
# # 保持低维度（形状【shape】中的最右边的数字）满足mxn@nxk=mxk
# # 高维度的元素个数【size】保持不变
# print(g @ h)

# # 逆运算:必须是方阵，必须为浮点数,逆运算之后的形状和原始形状一致
# a = torch.tensor([[[3, 2], [1, 2]], [[3, 2], [1, 2]]], dtype=torch.float32)
# b = torch.inverse(a)
# print(b)

# # 伪逆：当不满足方阵的条件时需要求他的伪逆
# a = torch.tensor([[3, 2], [1, 2], [3, 2]], dtype=torch.float32)
# # 求转置
# print(a.T)
# # 乘以转置变成方阵
# print(a.T @ a)
# # 求变成方阵后的逆
# # 再与该矩阵的转置叉乘，这就叫做a的伪逆
# # 为最小二乘法的铺垫
# print(torch.inverse(a.T @ a) @ a.T)

# # 张量中的统计函数
# a = torch.tensor([[[3, 2], [4, 2]], [[3, 4], [1, 3]]], dtype=torch.float32)
# # 同一维度的元素相加,会降低原始张量的一个维度
# print(a.sum())
# print(a.sum(dim=0))
# print(a.sum(dim=1))
# print(a.sum(dim=2))
#
# print(a.sum(1).sum(0))
#
# # 平均值
# print(a.mean())
# print(a.mean(2))
# # 标准差，用来评判数据与平均值之间的距离，也就是数据的离散程度
# print(a.std())
# # 将不符合正态分布的数据转化为正态分布
# # 样本平均值为0，并且标准差为1即符合正态分布
# b = (a - a.mean()) / a.std()
# print(b.mean(), b.std())

# # 最值
# # 索引说明最值取自当前维度【可以简单的判断，“[”数目一致张量为同一个维度的】的哪一个索引位置
# # indices=tensor([[1, 0],
# #         [0, 1]]))
# # print(a.max(1))
# print(a.argmax(1))
# print(a.min())


# # 改变张量的形状
# a = torch.tensor([[[3, 2, 2], [4, 2, 2]], [[3, 4, 2], [1, 3, 2]]], dtype=torch.float32)
# print(type(a))
# print(a.shape)
# print(a.reshape(1, 2, 6, 1))
# # 对轴进行交换来改变张量的形状
# print(a.permute(1, 2, 0))
