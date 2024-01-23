import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms


class dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = torch.Tensor(np.array(self.dataset[index].split(".")[1:5], dtype=np.float32) / 300)
        img_path = os.path.join(self.path, self.dataset[index])
        img = Image.open(img_path)
        img_data = torch.Tensor(np.array(img) / 255 - 0.5)
        # img_data = transforms.ToTensor()(img)
        return img_data, label


if __name__ == '__main__':
    mydata = dataset("data")
    dataloader = DataLoader(dataset=mydata, batch_size=20, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        print(img.shape)
        print(label.shape)
        # print(img)
        print(label)

        # x = img[0].numpy()
        # y = label[0].numpy()
        # img_data = np.array((x+0.5)*255,dtype=np.uint8)
        # img = Image.fromarray(img_data,"RGB")
        # draw = ImageDraw.Draw(img)
        # draw.rectangle(y*300,outline="red")
        # img.show()
