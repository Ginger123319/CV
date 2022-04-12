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
        self.sub_dir = "train_pic" if is_train else "test_pic"
        self.names = os.listdir(os.path.join(root, self.sub_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        filename = self.names[index]
        li = filename.split(".")[1:5]
        # 注意：标签归一化处理
        label = torch.Tensor(np.array(list(map(int, li)))) / 300
        img = Image.open(os.path.join(self.path, self.sub_dir, filename))
        img = transforms.ToTensor()(img)
        return img, label


if __name__ == '__main__':
    data = MyData("../../source/target_detection", is_train=False)
    print(data[0])
