from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np


class CIFARDataset(Dataset):
    def __init__(self, root, is_train=True):
        super(CIFARDataset, self).__init__()
        self.data = []
        sub_dir = "train" if is_train else "test"
        # 遍历目录，将图片路径和标签组成元组添加到dataset中
        # 注意都是字符串类型
        # 每一步都要进行测试
        for img_name in os.listdir(f"{root}/{sub_dir}"):
            img_path = f"{root}/{sub_dir}/{img_name}"
            tag = 0 if img_name[0] == "0" else 1
            self.data.append((img_path, tag))
        # print(self.dataset[5000])
        # print(tag)
        # print(type(img_name[0]))
        # print(type(img_name))
        # print(img_path)
        # break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index][0]
        img_data = cv2.imread(img_path) / 255
        # # <class 'numpy.ndarray'>
        # print(type(img_data))
        # # (100, 100, 3)
        # print(img_data)
        tag = self.data[index][1]
        return np.float32(img_data.transpose(2, 0, 1)), np.float32(tag)


if __name__ == '__main__':
    dataset = CIFARDataset(r"..\..\source\cat_dog\cat_dog", False)
    # print(dataset.__len__())
    print(dataset[0])
    # 每次取数据的时候会通过对象调用getitem方法
    dataLoader = DataLoader(dataset, batch_size=10, shuffle=True)
    # print(dataset.data[0][0])
    # img = cv2.imread(dataset.data[0][0])
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    for i, (x, y) in enumerate(dataLoader):
        print(i)
        print(x.shape)
        print(y)
