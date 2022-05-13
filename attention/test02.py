import torch
from torch import nn

torch.random.manual_seed(1)

a1 = torch.randn(2, 3)
a2 = torch.randn(2, 3)

b = torch.randn(3, 4)
# print(a1 @ b1)
# print(a2 @ b2)
#
a = torch.stack([a1, a2])
#
# print(a @ b)

net = nn.Linear(3, 4, bias=False)
# net.weight.data = b.T
print(net(a2))
