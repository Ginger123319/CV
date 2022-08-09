import random

import torch

# for _ in range(10):
#     # x, y = [random.randint(0, 150 - 1) for _ in range(2)]
#     # print(x, y)
#     print(random.randint(0, 150))
#     # break
# # 当无法进行广播，但是想进行相加的，需要进行repeat操作【前提是对应维度上元素个数（size）存在倍数关系】
m = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
n = torch.tensor([[3, 2], [1, 2], [3, 2], [3, 4]])
# repeat 在1轴上重复两次，每个元素复制一次[3,2]->[3,3,2,2]
# numpy中为np.tile(axis,repeat)函数，功能一致
print(n.shape)
o = torch.repeat_interleave(n, 2, 1)
print(o.shape)
# print(o[0, 0])
# print(m + o)
