import torch
import numpy as np


# 计算IOU交并比
def iou(box, boxes):
    # 计算所有矩形的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 计算交集矩形的坐标、左上角和右下角
    # 计算交集矩形的面积
    # 计算并集的面积，所有矩形面积总和减去交集的面积
    # 计算交并面积之比

    return 0


if __name__ == '__main__':
    data = np.array([1, 1, 5, 5])
    print(data)
