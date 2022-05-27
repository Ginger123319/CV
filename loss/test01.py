import torch

t = torch.tensor([1, 1, 0], dtype=torch.long)

f = torch.tensor([12, 13, 7])

center = torch.randn(2, 3)

h = torch.histc(t.float(), 2, min=0, max=1)
n = h[t]
print(n)

# print(center)
#
# print(center[t])
