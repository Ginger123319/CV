import torch

input = torch.arange(180).reshape(1, 2, 2, 3, 15)

mask = input[..., 0] > 6
print(mask)
print("mask:===>", mask.shape)
idxs = mask.nonzero()

print(idxs)
print(idxs.shape)
# # # print(idxs[:, 0].shape)
# print(input[mask].shape)
