import torch

a = torch.randn(2, 44)

b = a.reshape(2, 4, 11)
torch.softmax(b, -1)
