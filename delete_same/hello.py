import torch
from torch.version import cuda

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
