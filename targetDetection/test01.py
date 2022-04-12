import numpy as np

# li = ['1', '118', '55', '207', '144', 'png']
# x = 30000
# li[0] = str(x)
# print(li)
# print(type(str(x)))
import torch


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
    print(f"x_1 is {x_1}")
    print(x_2 - x_1)

    inv = w * h
    iou = inv / (area_1 + area_2 - inv)
    return iou


if __name__ == '__main__':
    data1 = torch.Tensor([[1, 1, 3, 3], [2, 2, 4, 4]])
    data2 = torch.Tensor([[2, 2, 3, 4]])
    data3 = data2.squeeze()
    print(data3)
    # print(iou(data2, data1))
