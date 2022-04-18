import torch

_cls = torch.randn(1,1,6,7)
print(_cls)
cls= _cls[0][0]
print(cls)
mask = torch.gt(cls, 0.6)
print(mask)
idxs = torch.nonzero(mask)
print(idxs)
# print(cls.shape)
# print(mask.shape)
print(idxs.shape)
# print(mask)
# print(idxs)
# for idx in idxs:
#     print(cls[idx[0], idx[1]])


