# 图像侦测中的注释

import torch
import numpy as np

# 例子1----torch.nonzero(torch.gt(cls, 0.6))

cls = torch.Tensor([[1,0.2,0.3,3],[1,0.2,0.3,3]])
idxs = torch.nonzero(torch.gt(cls, 0.6))

print(torch.gt(cls, 0.6))
print(idxs) #返回每个元素所对应的索引（行索引，列索引）[[0 0] [0 3] [1 0][1 3]]
print("---------------------------------")
print(idxs[0]) #1st值的索引，[0][0]
print(idxs[1]) #2nd值的索引，[0][3]
print(idxs[2]) #3rd值的索引，[1][0]
print(idxs[3]) #4th值的索引，[1][3]


# 例子2----np.stack：堆叠
a = np.array([1,2])
b = np.array([3,4])
c = np.array([5,6])

list = []

list.append(a)
list.append(b)
list.append(c)

print(list) # [array([1, 2]), array([3, 4]), array([5, 6])]

d = np.stack(list) #在0轴上堆叠，也可指定轴
print(d) #[[1 2][3 4][5 6]]



# 例子3---np.where(cls>0.97):返回两值，1st为0轴索引，2nd为1轴索引，共同决定元素位置
cls = torch.Tensor([[1,2],[0.8,1],[1,1]])

idxs,_ = np.where(cls>0.97)
print(cls>0.97) #tensor([[1, 1],[0, 1],[1, 1]], dtype=torch.uint8)
print(np.where(cls>0.97)) #(array([0, 0, 1, 2, 2], dtype=int64), array([0, 1, 1, 0, 1], dtype=int64))
print(idxs,_) #0轴索引：[0 0 1 2 2]；1轴索引：[0 1 1 0 1]；例如2:的索引（0,1）:0轴位0,1轴为1
