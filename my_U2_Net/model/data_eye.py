# coding=gbk
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np


class DataEye(Dataset):
    def __init__(self, root, is_train=True):
        super().__init__()
        self.flag = is_train
        self.root = root
        # �ֱ�ȡ��ѵ������ǩ�Ͳ��Լ���ͼƬ�ļ����б�
        if self.flag:
            self.train_tag_names = os.listdir(os.path.join(root, r"training\1st_manual"))
        else:
            self.test_image_names = os.listdir(os.path.join(root, r"test\images"))

    def __len__(self):
        if self.flag:
            return len(self.train_tag_names)
        else:
            return len(self.test_image_names)

    def __getitem__(self, index):
        # ��ͼ����д���
        transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)])
        # �����ļ����õ�������Ҫ��ͼƬ��·��
        if self.flag:
            train_tag_path = os.path.join(self.root, r"training\1st_manual")
            train_img_path = os.path.join(self.root, r"training\images")
            train_tag_name = self.train_tag_names[index]
            train_img_name = train_tag_name[0:3] + "training.tif"
            train_img = cv2.imread(os.path.join(train_img_path, train_img_name))
            train_tag = Image.open(os.path.join(train_tag_path, train_tag_name))
            # train_tag.show()
            # train_tag = np.array(train_tag)
            # cv2.imshow("img", train_img)
            # cv2.waitKey(0)
            seed = np.random.randint(10000)
            # torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            train_img = transforms.ToPILImage()(train_img)
            img = transform(train_img)
            torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            tag = transform(train_tag)
            # ��ҪչʾͼƬ����Ҫ��transform�е�ToTensor�رգ�
            # ��Ϊ�����й�һ���ͻ��ᣬ������RGBֵ��HWC��ʽ
            # train_img_data = np.array(img)
            # train_tag_data = np.array(tag)
            # cv2.imshow("img", train_img_data)
            # cv2.waitKey(0)
            # cv2.imshow("tag", train_tag_data)
            # cv2.waitKey(0)
            return img, tag

        else:
            test_img_name = self.test_image_names[index]
            test_img_path = os.path.join(self.root, r"test\images")
            test_img = cv2.imread(os.path.join(test_img_path, test_img_name))
            return transforms.ToTensor()(test_img)


if __name__ == '__main__':
    train_data = DataEye(r"..\..\..\source\EYE_DATA", is_train=True)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    # print(train_data[0])
    for i, (img1, tag1) in enumerate(train_loader):
        print(i)
        print(img1.shape)
        print(tag1.shape)
