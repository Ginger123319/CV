# stack的用法

import numpy as np

a = np.array([1,2])
b = np.array([3,4])
c = np.array([5,6])

ls = []
ls.append(a)
ls.append(b)
ls.append(c)

print(ls)
print(np.stack(ls))
print(np.stack(ls,axis=1))
