import numpy as np

a = np.array([[[0.3],[0.8],[0.6],[0.2],[0.7],[0.9],[0.5]]])
# a = a[:,0]
print(a.shape)
# print(a>0.4)
_,idxs, _ = np.where(a > 0.4)
print(idxs)
# print(np.where(a > 0.4))