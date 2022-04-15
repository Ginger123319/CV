import torch
import numpy as np


# # 计算IOU交并比
# def iou(box, boxes):
#     # 计算所有矩形的面积
#     box_area = (box[2] - box[0]) * (box[3] - box[1])
#     boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     # 计算交集矩形的坐标、左上角和右下角
#     # 计算交集矩形的面积
#     # 计算并集的面积，所有矩形面积总和减去交集的面积
#     # 计算交并面积之比
#
#     return 0
def iou(arr_1, arr_2):
    area_1 = (arr_1[:, 2] - arr_1[:, 0]) * (arr_1[:, 3] - arr_1[:, 1])
    area_2 = (arr_2[:, 2] - arr_2[:, 0]) * (arr_2[:, 3] - arr_2[:, 1])
    # 交集面积
    x_1 = np.maximum(arr_1[:, 0], arr_2[:, 0])
    y_1 = np.maximum(arr_1[:, 1], arr_2[:, 1])
    x_2 = np.minimum(arr_1[:, 2], arr_2[:, 2])
    y_2 = np.minimum(arr_1[:, 3], arr_2[:, 3])
    w = np.maximum(0, x_2 - x_1)
    h = np.maximum(0, y_2 - y_1)
    # print(f"x_1 is {x_1}")
    # print(x_2 - x_1)

    inv = w * h
    result_iou = inv / (area_1 + area_2 - inv)
    return result_iou


if __name__ == '__main__':
    data1 = torch.Tensor([[1, 1, 3, 3], [2, 2, 4, 4]])
    data2 = torch.Tensor([[1, 1, 2, 4], [2, 2, 3, 4]])
    print(iou(data2, data1))
