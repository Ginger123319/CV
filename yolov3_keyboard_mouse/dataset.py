import sys

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import pic_resize
import torchvision
import numpy as np
import cfg
import os

from PIL import Image
import math

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self, label_path, img_path):
        with open(label_path) as f:
            self.dataset = f.readlines()
            self.img_path = img_path
            # print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 用字典存储三个尺度的标签4，8，16
        labels = {}
        # 取出字符串列表中的一条数据
        line = self.dataset[index]
        # 将这个字符串按空格切分为字符列表，会忽略掉\n
        strs = line.split()
        # print(strs)
        # exit()
        # _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        # print(_img_data.size)

        # _img_data = _img_data.resize((416, 416))  # 此处要等比缩放
        _img_data, ratio = pic_resize(os.path.join(self.img_path, strs[0]))
        # print("ratio:", ratio)
        img_data = transforms(_img_data)
        # print(img_data.shape)
        # exit()
        _boxes = np.array([float(x) for x in strs[1:]])
        # _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)
        # print(len(boxes))
        # 缩放标签,是否有更好的方式或者更加简洁的写法
        for i in range(len(boxes)):
            boxes[i][1:] = boxes[i][1:] * ratio

        index = 0
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            # 创建4尺度的标签，此处labels是个字典
            # 键值对（4：标签张量）
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            # print(labels[feature_size].shape)
            # exit()
            for box in boxes:
                cls, cx, cy, w, h = box
                # 计算偏移量，以及中心点在标签上的索引位置
                # math.modf先返回小数部分，再返回整数部分
                # feature_size / cfg.IMG_WIDTH=(4,8,16)/416
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                # print(cx_index,cx_offset)
                # exit()

                # 取出三个尺寸的建议框
                for i, anchor in enumerate(anchors):
                    # 取出每个（i：0，1，2）建议框的面积
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    # 计算宽高的偏移量
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # 计算目标框和建议框的IOU
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    index += 1
                    # 向目标中心点对应到标签上的索引位置填充一个IOU，四个偏移量，以及分类标签
                    # 标签的形状是HWC，所以cy_index在前面
                    # i表示向第几个框中填写这六个标签值
                    # np.log(p_w)，因为宽高的偏移量只能是正数，取对数后，标签就有正有负，更利于模型训练
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), int(cls) + 1])
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    data = MyDataset(cfg.train_path, cfg.img_path)
    # print((data[0][3].permute(1, 2, 0).numpy() * 255).shape)
    # img = np.uint8(data[0][3].permute(1, 2, 0).numpy() * 255)
    # print(img.dtype)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    loader = DataLoader(data, batch_size=32, shuffle=True)
    for i, (x1, x2, x3, d) in enumerate(loader):
        print(i)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(d.shape)
    # print(data[1][0].shape)
    # print(data[1][0][..., -1] > 0)
    # print(np.sum(np.sum((data[1][0][..., -1] > 0), axis=-1) == 3))
    # for i in range(14):
    #     print(data[i][0].shape)
    # 打印所有的IOU
    # print(data[2][0][..., -1])
    # print(data[0][1].shape)
    # print(data[0][2].shape)
    # print(data[0][3].shape)
    # myDataset = MyDataset(cfg.train_path, cfg.img_path)
    # train_loader = DataLoader(myDataset, batch_size=3, shuffle=False)
    # for target_4, target_8, target_16, img_data in train_loader:
    #     print(target_4.shape)
    #     # print(torch.sum((target_4[..., -1] > 0), dim=-1))
    #     print(torch.sum(torch.sum((target_4[..., -1] > 0), dim=-1) == 3))
    #     exit()
