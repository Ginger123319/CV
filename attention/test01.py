import torch

a = torch.randn(3, 4, 5)

# b = a.permute(0, -1, -2)
# print(b.shape)
b = torch.transpose(a,-1,-2)
print(b.shape)
