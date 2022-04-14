import numpy as np

bs = np.array([[1,1,10,10,40],[1,1,9,9,10],[9,8,13,20,15],[6,1,18,17,13]])
a = bs[(-bs[:,4]).argsort()]
print(a)
# print(bs[:,4])
# print((-bs[:,4]))
# print((-bs[:,4]).argsort())

