import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math

LABEL_FILE_PATH = "data/Parse_label.txt"  # 标签数据地址
IMG_BASE_DIR = "data/"  # 数据总地址

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])  # 对数据进行处理


def one_hot(cls_num, i):  # 做一个onehot分类函数
    b = np.zeros(cls_num)  # 编一个类别数的一维o数组
    b[i] = 1.  # 在指定的位置填充1
    return b


class MyDataset(Dataset):  # 做数据类

    def __init__(self):  # 定义初始化函数
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()  # 读取标签文件里面的所有数据
            # print(self.dataset)

    def __len__(self):
        return len(self.dataset)  # 获取数据长度

    def __getitem__(self, index):  # 用__getitem__方法一个一个取出数据
        labels = {}  # 创建一个空的字典
        line = self.dataset[index]  # 根据索引获取每张图片的数据
        # print(line)
        strs = line.split()  # 将数据分隔开
        # print(strs)
        a = os.path.join(IMG_BASE_DIR, strs[0])
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))  # 打开图片数据
        # print(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)  # 对数据进行处理，转Tosor
        # _boxes = np.array(float(x) for x in strs[1:])#将str列表1：后面的所有数据转成float类型
        # print(_boxes)
        _boxes = np.array(list(map(float, strs[1:])))  # 也可以用这种方式转成float类型
        boxes = np.split(_boxes, len(_boxes) // 5)  # 将_boxes列表中的元素5等分并赋值给boxes，
        # 这里为5的原因是因为每个框有5个标签，除以5就可以把每个框分开，拿到框的数量。

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():  # 循环标签框，并将三种建议狂，和
            # 每种输出的框分别负责给两个变量，循环的目的是输出有3中特征图，因此也需要三种标签
            # print(feature_size,anchors)
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))  # 在空的字典中以
            # feature_size为键形状为shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))的0矩阵为值得字典，
            # feature_size, feature_size为输出特征图尺寸大小 3为3个建议狂，5 + cfg.CLASS_NUM为自信度、两个中心点坐标，两个偏移量+类别数

            for box in boxes:  # 循环框的个数
                cls, cx, cy, w, h = box  # 将每个框的数据组成的列表解包赋值给对应变量
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)  # 计算中心点的在每个格子x方向的偏移量和索引,
                # 中心点x的坐标乘以特征图的大小再除以原图大小，整数部分作为索引，小数部分作为偏移量，原本是cx/(图片总的大小/特征图大小）
                # ，展开括号就等于cx乘以特征图大小除以图片总的大小。   y方向同理
                # math.modf 方法返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示。
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)  # 计算中心点的y方向的偏移量和索引
                for i, anchor in enumerate(anchors):  # 循环3种建议框，并带索引分别赋值给i和anchor
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]  # 每个建议框的面积，面积计算在宁一个模块
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    # anchor[0]、 anchor[1]为建议框放的宽和高，w / anchor[0], h / anchor[1]代表建议框和是实际框在w和h方向的偏移量，并赋值
                    # print(anchors[1])
                    # print(p_w)
                    # print(p_h)
                    p_area = w * h  # 实际标签框的面积
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)  # 计算建议框和实际框的iou，用最小面积除以最大面积
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                         *one_hot(cfg.CLASS_NUM, int(cls))])  # 10,i
                    # 根据索引将[iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))]
                    # 填入对应的区域里面作为标签，表示该区域有目标物体，其他地方没有就为0，
                    # labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))，
                    # 由于labels的形状使这种，因此要将5 + cfg.CLASS_NUM填入对应位置，因此要在3个框的内部的对应格子里面，
                    # 因此feature_size, feature_size, 3要作为索引去索引对应位置，feature_size, feature_size代表特征图大小，
                    # 也代表格子数，因此针对一个框的目标位置索引应该为目标所在格子的位置和所在框的序号，
                    # 因此这里才会写成labels[feature_size][int(cy_index), int(cx_index), i]

        return labels[13], labels[26], labels[52], img_data  # 将三种标签和数据返回给调用方


if __name__ == '__main__':
    x = one_hot(4, 2)
    # print(x)
    data = MyDataset()
    dataloader = DataLoader(data, 3, shuffle=True)
    # for i,x in enumerate(dataloader):
    #     print(x[0].shape)
    #     print(x[1].shape)
    #     print(x[2].shape)
    #     print(x[3].shape)
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13.shape)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)
