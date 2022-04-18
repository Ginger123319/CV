# 掩码----masked_select

import torch

a =  torch.Tensor([[1,2,3,4,5]])

print(a<4)
print(torch.lt(a,4)) # lt gt eq le ge：小于 大于 等于 小于等于 大于的等于

# 两者结果相同
print(a[a<4])
print(torch.masked_select(a, a<4))

# print(torch.nonzero(a<4))