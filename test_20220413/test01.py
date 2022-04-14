import torch
from torch import nn

layer = nn.Conv2d(3,5,12,2)
# layer2 = nn.Linear(5*106*106,10)

x = torch.randn(1,3,221,221)
y = layer(x)
# y = y.reshape(-1,5*106*106)
# y = layer2(y)
print(y.shape)