import torch

cls = torch.Tensor([0,1,2,0,1,0])
off = torch.Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6]])
# print(cls<2)
# print(cls[cls<2])
# #
# print(torch.lt(cls,2))
# print(torch.masked_select(cls,torch.lt(cls,2)))
offset_mask = cls>0
print(cls>0)
print(torch.gt(cls,0))
print(off[cls>0])
offset_index = torch.nonzero(offset_mask)
offset_index_ = torch.nonzero(offset_mask)[:, 0]
# print(offset_mask)
print(offset_index)
print(offset_index_)
# print(off[offset_mask])
print(off[offset_index_])

