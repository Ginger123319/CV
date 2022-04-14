import numpy as np
import torch

li = ['1', '118', '55', '207', '144', 'png']
x = 30000
li[0] = str(x)
# print(li)
# print(type(str(x)))
# 字符列表转为整型列表
li = ['1', '118', '55', '207', '144']
# li = list(map(int, li))
# 张量都具有广播的特性
li = np.array(li, dtype=np.float32())
# print(li * 0.5)
li = torch.Tensor(li)
ratio = 0.5
# print(li * ratio)
# print(li)


# if __name__ == '__main__':
data1 = torch.Tensor([[1, 1, 3, 3], [2, 2, 4, 4]])
data2 = np.array([2, 2, 3, 4])
data3 = np.array([1, 0, 1, 0])
index = np.where(data2 > 2)
print(index)
print(type(index))
a = data2[index]
# b = data1[index]
# print(b)
# print(index)
print(a)

data2 = data2 > 2
print(data2)
temp = np.float32(data2)
temp1 = np.mean(np.float32(temp.equal(data3)))
print(temp1)

# print(len(a))
# #     data3 = data2.squeeze()
# #     print(data3)
#
# data2 = np.array([])
# print(len(data2))
