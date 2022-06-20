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

# # 使用掩码取张量，一般来说掩码形状应该和张量形状一致,或者是和0轴形状一致的向量
# mask = x[:, 0] > 2
# print(mask.shape)
# print(mask)
# print(x[mask])

# 使用nonzero获取的索引，一般不会用在原始张量取值的操作上
# 只是通过索引分析出图片所在的批次，HWC之类的信息；获取和索引相关的信息
mask = x > 2
mask_index = mask.nonzero()
print(mask_index)

# # 广播的形式就是将同样的张量复制n份，然后在1所在的维度cat起来
# print(mask[:, None] * x)
# print(torch.cat([mask[None, :], mask[None, :]]))
# print(torch.cat([mask[:, None], mask[:, None]],dim=-1))
