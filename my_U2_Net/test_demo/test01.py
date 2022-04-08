import random
import numpy as np

# a = random.random()
# print(a)
# img = np.random.randn(2,2,3)
# print(img)
# print()
# image = img[::-1]
# print(image)

x = np.arange(12).reshape([2,2,3])
print(x,"\n")
y = x[::-1]
print(y)
