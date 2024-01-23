from torch.utils.data import Dataset, DataLoader
import torch
import os
import cv2
import numpy as np


class Flower(Dataset):
    def __init__(self, path):
        train_data = []
        test_data = []
        train_target = []
        test_target = []
        import random

        for root, sub_dir, files in os.walk(path):
            if len(files) > 0:
                cls = 0
                print(root)
                random.seed(1)
                data = random.sample([i for i in range(len(files))], int(0.1 * len(files)))
                for i in range(len(files)):
                    # imread的速度很慢
                    img = cv2.imread(os.path.join(root, files[i]))
                    if i in data:
                        test_data.append(img)
                        test_target.append(cls)
                    else:
                        train_data.append(img)
                        train_target.append(cls)
                cls += 1
        print(len(test_target), len(test_data), len(train_data), len(train_target))

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    path = r'D:\Python\source\flowers'
    flower = Flower(path)