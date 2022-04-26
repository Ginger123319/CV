import numpy as np
import cv2 as cv
import os

# 等比例缩放为416x416
pic_root = r"C:\Users\liev\Pictures\Camera Roll\img"
target_len = 416


def pic_resize(path):
    img_data = cv.imread(path)
    # cv.imshow("pic", img_data)
    # cv.waitKey()
    # cv.destroyAllWindows()
    h = img_data.shape[0]
    w = img_data.shape[1]
    max_len = max(h, w)
    # 计算缩放比例
    ratio = target_len / max_len
    h = int(h * ratio)
    w = int(w * ratio)
    # print(h, w)
    # 按比例缩放
    dst = cv.resize(img_data, (w, h))
    # 给缩放后的图片加黑边，在下面或者右边添加
    # 不需要分类，有一条边为416，计算结果就为0
    dst = cv.copyMakeBorder(dst, 0, 416 - h, 0, 416 - w, cv.BORDER_CONSTANT, value=[0, 0, 0])
    # print(dst.shape)
    # cv.imshow("pic", dst)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # 返回处理后的图片数据以及缩放比例
    return dst, ratio


for pic_name in os.listdir(pic_root):
    pic_path = os.path.join(pic_root, pic_name)
    print(pic_resize(pic_path))
    print(pic_name)
    exit()
