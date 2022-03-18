import torch
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
b = a[360:720:, 640:1280]
print(b.shape)
res_img = im.fromarray(b)
res_img.show()
