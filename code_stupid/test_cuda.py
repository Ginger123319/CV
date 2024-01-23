import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

listA = ['A', 'B', 'C']
listB = ['C', 'A', 'B']
listA.sort()
listB.sort()
print(listA != listB)
# print(torch.zeros(30, 20).dtype)
exit()
print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.get_device_name(0))

print(torch.cuda.current_device())

print(torch.cuda.get_device_properties(0))

dist.init_process_group()
