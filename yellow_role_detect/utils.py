import numpy as np


def iou(arr_1, arr_2):
    area_1 = (arr_1[2] - arr_1[0]) * (arr_1[3] - arr_1[1])
    area_2 = (arr_2[2] - arr_2[0]) * (arr_2[3] - arr_2[1])
    # 交集面积
    x_1 = np.maximum(arr_1[0], arr_2[0])
    y_1 = np.maximum(arr_1[1], arr_2[1])
    x_2 = np.minimum(arr_1[2], arr_2[2])
    y_2 = np.minimum(arr_1[3], arr_2[3])

    inv = (x_2 - x_1) * (y_2 - y_1)
    iou = inv / (area_1 + area_2 - inv)
    return iou
