# nonzero的用法

import torch
import numpy as np

a = torch.Tensor([[1],[3],[5],[7]])
b = torch.Tensor([[1,2,3,4],[2,3,4,5],[5,6,7,8],[7,8,9,10]])
# print(a>3)
# print(a[a>3])

print(a[torch.nonzero(a>3)])
print(torch.nonzero(a>3))
print(b[torch.nonzero(a>3)[:,0]])
