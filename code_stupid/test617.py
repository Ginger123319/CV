import torch

# 在每次重新运行程序时，同样的随机数生成代码得到的是同样的结果。
# 如果需要同一段程序中每次随机生成的数字都一样，需要将每次随机生成之前都将随机数种子固定
random_seed = 101
torch.manual_seed(random_seed)
x = torch.randint(1, 10, (5, 5))
print(x)
# 切片start:end:step
# print(x[..., ::2])
# torch.manual_seed(random_seed)
# print(torch.rand(1))
# 使用掩码取张量，一般来说掩码形状应该和张量形状一致，或者能够广播为张量的形状（5和5*5；注意广播的方式）
mask = x[:, 0] > 2
print(torch.repeat_interleave(mask, 5))
mask_index = mask.nonzero()
print(mask_index)
print(mask.shape)
print(mask)
print(x[mask])
print(x[mask_index].shape)
