import os
from PIL import Image
import numpy as np
import torch

path = 'img'  # 地址
img_name = os.listdir(path)  # 获取该地址下的文件名称
# print(img_name)

for name in img_name:
    img = Image.open(os.path.join(path, name))  # 打开图片
    w, h = img.size[0], img.size[1]  # 获取图片的宽高
    a = np.maximum(w, h)  # 获取wh中的最大值
    # print(a)
    # print(img.size)
    goal_img = Image.new('RGB', (a, a), color=(0, 0, 0))  # 创造一个黑板图片
    goal_img.paste(img, (int((a - w) / 2), int((a - h) / 2)))  # 将图片贴到目标图片上
    images = goal_img.resize((416, 416))  # 改变图片大小
    print(images.mode)  # 查看图片类型
    # images.save(os.path.join('images/',name))
    # print(images.size)
    # goal_img.show()
