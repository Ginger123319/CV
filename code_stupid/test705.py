import torch
from torch import nn
import math

a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.Tensor([[1, 2, 3], [1, 5, 6], [1, 8, 9]])
c = torch.Tensor([[1, 2, 3]])
# print(a * c)
# print(a * b)
# print(torch.matmul(a, b))
x_input = torch.tensor([[0.3, 0.6]], requires_grad=True)
target = torch.tensor([1])
ce = nn.CrossEntropyLoss()
# print(math.log(0.3), math.log(0.6), math.log(0.1))
# first = -x_input[0][2]
# second = 0
# for i in range(1):
#     for j in range(3):
#         second += math.exp(x_input[i][j])
# res = first + math.log(second)
# print(first)
# print(res)
print(ce(x_input, target))
# print(torch.exp(-res))
print(-math.log(0.2584))
# 测试数据经过softmax处理后再经过多分类交叉熵公式计算的结果和pytorch中自带的交叉熵函数计算结果就是一致的
print(torch.softmax(x_input, dim=-1))

# import torch
# import torch.nn as nn
# import math
#
# criterion = nn.CrossEntropyLoss()
# # output = torch.randn(1, 5, requires_grad=True)
# output =  torch.tensor([[0.3, 0.6, 0.1]],requires_grad=True)
# label = torch.empty(1, dtype=torch.long).random_(3)
# loss = criterion(output, label)
#
# print("网络输出为5类:")
# print(output)
# print("要计算label的类别:")
# print(label)
# print("计算loss的结果:")
# print(loss)
#
# first = 0
# for i in range(1):
#     first = -output[i][label[i]]
# second = 0
# for i in range(1):
#     for j in range(3):
#         second += math.exp(output[i][j])
# res = 0
# res = (first + math.log(second))
# print(first)
# print("自己的计算结果：")
# print(res)
