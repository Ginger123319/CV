import torch
from torch import Tensor
import numpy as np
from PIL import Image as im

image = im.open("7505.png")
# image.show()
a = np.array(image)
print(a)

# b = torch.from_numpy(a)
# print(b)

print(a.shape)
# # 将一张图片均分为四张图片
# b = a.reshape(2, 540, 2, 960, 3)
# print(b.shape)
# c = b.transpose(0, 2, 1, 3, 4)
# print(c.shape)
# result = c.reshape(4, 540, 960, 3)
# for i in range(4):
#     res_img = im.fromarray(result[i])
#     res_img.show()

# 将图片缩放为原来的1/4，对高和宽使用切片，步长为2
# b = a[::2, ::2]
# 取图片的中间部分
# b = a[360:720:, 640:1280]
# print(b.shape)
# res_img = im.fromarray(b)
# res_img.show()


# 使用torch实现一次
# 将numpy数组转为tensor张量
b = torch.from_numpy(a)
print(b.shape)

# # 将图片旋转九十度，交换前面两个轴
# c = b.permute(1, 0, 2)
# res_img = im.fromarray(Tensor.numpy(c))
# print(type(res_img))
# res_img.show()

# # 将图片切割为均匀的四部分，对图片进行形状变换。需要将高和宽都切割成原来的一半
# c = b.reshape(2, 540, 2, 960, 3)
# print(c.shape)
# # 交换0轴和2轴的位置
# d = c.permute(0, 2, 1, 3, 4)
# print(d.shape)
# # 再改变交换后的张量
# e = d.reshape(4, 540, 960, 3)
# for i in e:
#     print(i.shape)
#     res_img = im.fromarray(Tensor.numpy(i))
#     res_img.show()

# 取图片的中间部分：对张量进行切片
# print(b[360:720, 640:1280].shape)
