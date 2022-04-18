# 按索引排序，并取值

import numpy as np

a = np.array([[5,3],[3,2],[1,8],[2,4]])
index = (-a[:,1]).sort()
b = a[index]

print(b)
