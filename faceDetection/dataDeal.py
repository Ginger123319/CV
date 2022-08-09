import os
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw


class MyData(Dataset):
    def __init__(self, root, is_train=True):
        super().__init__()
        self.path = root
        self.flag = is_train
        self.dataset = []
        self.train_dataset = []
        self.test_dataset = []
        # print(self.path)
        tag_path = os.path.join(self.path, r"Anno\list_bbox_celeba.txt")
        # print(tag_path)
        img_pre_path = os.path.join(self.path, r"img\img_celeba\img_celeba")
        counter = 1
        with open(tag_path, "r") as a_file:
            a_file.readline()
            a_file.readline()
            for line in a_file:
                line = line.split()

                filename = line[0]
                img_path = os.path.join(img_pre_path, filename)
                # print(img_path)
                li = list(map(int, line[1:]))
                # print(li)
                li[2] += li[0]
                li[3] += li[1]
                # print(li)
                if os.path.exists(img_path):
                    self.dataset.append((img_path, li))
                # print(self.dataset)
                if counter > 24000:
                    break
                counter += 1
        if self.flag:
            self.train_dataset = self.dataset[0:20000]
        else:
            self.test_dataset = self.dataset[20000:]

    def __len__(self):
        if self.flag:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.flag else self.test_dataset
        img_path = dataset[index][0]
        tag = dataset[index][1]
        # print(f"org tag is {tag}")
        # print(tag)
        img = Image.open(img_path)
        w, h = img.size
        x1 = tag[0] * (300 / w)
        y1 = tag[1] * (300 / h)
        x2 = tag[2] * (300 / w)
        y2 = tag[3] * (300 / h)
        # print(tag)
        label = np.array([x1, y1, x2, y2])
        # print(label)
        label_max = label.max()
        # print(label_max)
        # 注意：标签归一化处理
        label = torch.Tensor(label) / 300
        img = img.resize((300, 300))
        # draw = ImageDraw.Draw(img)
        # draw.rectangle(tag, outline="blue", width=3)
        # img.show()
        img = transforms.ToTensor()(img)
        return img, label


if __name__ == '__main__':
    data = MyData(r"D:\Python\source\FACE\celebA", is_train=True)
    # print(data[7])
    data_loader = DataLoader(data, batch_size=2, shuffle=False)
    # for epoch in range(1):
    #     for i, (img, label) in enumerate(data_loader):
    #         print(i)
    #         print(img)
    #         print(label)
    #         break
