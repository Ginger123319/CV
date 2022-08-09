import numpy as np

a = np.where(np.array([1,2,3,4,5,6])<4)
print(a)

b = []
c = np.array([1,2,3])

b.append(c)
b.append(c)
b.append(c)
b = np.stack(b)
print(b)