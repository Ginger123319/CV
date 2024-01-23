import torch
a = torch.randn(2,3,1,1)
b = a.squeeze_()
c = a.squeeze()
print(b is a)
