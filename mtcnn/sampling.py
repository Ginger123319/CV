# 创建数据集

from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


# 数据集
class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())  # 打开正样本标签文档，逐行读取，再添加至列表中
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)  # 数据集长度

    def __getitem__(self, index):  # 获取数据
        strs = self.dataset[index].strip().split(" ")  # 取一条数据，去掉前后字符串，再按空格分割
        # 标签：置信度+偏移量
        cond = torch.Tensor([int(strs[1])])  # []莫丢，否则指定的是shape
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        # 样本：img_data
        img_path = os.path.join(self.path, strs[0])  # 图片绝对路径
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)  # 打开-->array-->归一化去均值化-->转成tensor
        img_data = img_data.permute(2, 0, 1)  # CWH

        # print(img_data.shape) # WHC
        # a = img_data.permute(2,0,1) #轴变换
        # print(a.shape) #[3, 48, 48]：CWH

        return img_data, cond, offset


# 测试
if __name__ == '__main__':
    path = r"D:\Python\source\FACE\celebA\save_pic_label\48"  # 只以尺寸为48的为例
    dataset = FaceDataset(path)
    print(dataset[0])
    print(dataset[0][2].shape)
