# 菌落识别
import cv2
import numpy as np

path = r"..\..\source\opencv_pic\1.png"
path1 = r"..\..\source\opencv_pic\pre_mask.jpg"

img = cv2.imread(path)  # shape(3648, 5472, 3)

# 中值滤波（去除噪点）
img2 = cv2.medianBlur(img, 9)
gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# 自适应二值化（识别杂质）
# ret, thresh_const = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)
cv2.imwrite("thresh.png", np.hstack([thresh_adaptive]))

# 矩阵运算叠加原图（识别结果）
# 圆的坐标和半径
x, y, r = 2868.5, 1732.5, 1450


# 计算像素坐标是否在圆范围，返回布朗值
def judge(i, j):
    distance = np.sqrt((y - i) ** 2 + (x - j) ** 2)
    return distance < r


# 圆边界计算
border = np.fromfunction(judge, thresh_adaptive.shape)
# 叠加检测结果矩阵
border = np.where(thresh_adaptive > 0, border, False)

b, g, r = cv2.split(img2)
b = np.where(border, 255, b)
g = np.where(border, 0, g)
r = np.where(border, 255, r)

# 合并三个通道得到最终图
img_res = cv2.merge([b, g, r])
cv2.imwrite("result.png", img_res)


