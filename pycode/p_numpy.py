import numpy as np
import torch

# # 张量及相关操作
# # 张量有形状，与python的列表含义不同
# a = [[1, 2, "s"], [1, 2]]
# print(a[0][1])
# # 创建张量时对元素个数和类型有限制
# # ndarrays with different lengths or shapes)
# # b = np.array([[1, 2, 3], [1, 2]])
# b = np.array([[1, 2, 3], [1, 2, 3]])
# print(b, b.dtype)
#
# c = np.zeros((2, 3))
# print(c)
#
# # 修改张量
# print(b[1][2])
# print(b[1, 2])
# b[1][2] = 5
# print(b)
# # 张量切片，注意和列表切片区分开
# print(b[:, 1])
# print(b[1, :])

# 张量合并
d = torch.tensor([[1, 2], [3, 4], [3, 4]])
e = torch.tensor([[4, 5], [5, 6]])
g = torch.tensor([[[4, 5], [4, 5]], [[4, 5], [5, 4]]])
h = torch.tensor([[[4, 5], [4, 5]], [[4, 5], [5, 4]]])
print(d.shape)
# 当有多个轴的时候，在某个轴上进行合并，参数就写哪个轴
# 需要保证在其他轴上至少有一个是形状相同的
f = torch.cat((d, e), dim=0)
print(f)
print(g.shape)
# 张量的统计函数
print(g)

print(torch.randn(2))

