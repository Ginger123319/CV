import numpy as np

bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 1, 18, 17, 13]])
print(bs)
# 降序排序
flag = bs[:, -1]
print(flag)

bs = bs[np.argsort(-flag)]
print(bs)

test = bs[:, 2]
index = np.where(test < 13)
print(type(index))
print(test)
