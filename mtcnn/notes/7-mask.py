# 掩码

import numpy as np

a= np.array([8,2,7,5,1,4])

print(a<5)
print(a[a<5])

print(np.where(a<5))
print(a[np.where(a<5)])
