import torch
from torch.nn.functional import interpolate

test = torch.randn(1, 1, 3, 3)
result = interpolate(test, scale_factor=2, mode='nearest')
print(result.shape)
