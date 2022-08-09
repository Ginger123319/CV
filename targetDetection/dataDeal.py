import os
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):
    def __init__(self, root, is_train=True):
        super().__init__()
        self.path = root
        self.flag = is_train
        # self.sub_dir = "train_pic" if is_train else "test_pic"
        self.sub_dir = "train_pic_plus" if is_train else "test_pic_plus"
        self.names = os.listdir(os.path.join(root, self.sub_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        filename = self.names[index]
        # li = filename.split(".")[1:5]
        li = filename.split(".")[1:6]
        # 注意：标签归一化处理
        label = torch.Tensor(np.array(list(map(int, li))))
        # print(label)
        # label1：坐标；label2：分类概率
        label1 = label[0:4]/300
        label2 = label[-1]
        img = Image.open(os.path.join(self.path, self.sub_dir, filename))
        img = transforms.ToTensor()(img)
        # return img, label1
        return img, label1, label2


if __name__ == '__main__':
    data = MyData("../../source/target_detection", is_train=True)
    print(data[500])
