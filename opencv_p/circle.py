import cv2
import numpy as np
from cv2 import contourArea

path = r"..\..\source\opencv_pic\1.png"
path1 = r"..\..\source\opencv_pic\pre_mask.jpg"
img = cv2.imread(path)
img1 = cv2.imread(path1)
# print(img.shape, img1.shape)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
# 闭操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close_dst = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
# 开操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
open_dst = cv2.morphologyEx(close_dst, cv2.MORPH_OPEN, kernel, iterations=1)
# # print(mask.shape)
# 截取图片
cut_pic = cv2.bitwise_and(img, img, mask=open_dst)
# # print(cut_pic.shape)
# 中值滤波
cut_pic = cv2.medianBlur(cut_pic, 5)
cut_gray = cv2.cvtColor(cut_pic, cv2.COLOR_BGR2GRAY)
result_binary = cv2.adaptiveThreshold(cut_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 6)
# _, result_binary1 = cv2.threshold(cut_gray, 177, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
result_binary = cv2.erode(result_binary, kernel)
# # 开操作
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
# mask1 = cv2.morphologyEx(result_binary, cv2.MORPH_OPEN, kernel, iterations=1)
# # 礼帽操作，获取白噪
# result = result_binary - mask1
# 寻找二值图上的轮廓并将轮廓点画在原图上
contours, image = cv2.findContours(result_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dot = []
for elem in contours:
    if 2 < len(elem) < 85:
        dot.append(elem)
print(len(dot))
img_contour = cv2.drawContours(img, dot, -1, (0, 255, 0), 2)
# 使得图片自适应展示
cv2.namedWindow("win1", cv2.WINDOW_NORMAL)
cv2.imshow("win1", img_contour)
# cv2.namedWindow("img_contour", cv2.WINDOW_NORMAL)
# cv2.imshow("img_contour", result_binary)
# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.imshow("win", binary)
cv2.waitKey()
cv2.destroyAllWindows()
# print(len(contours))
# # 计算轮廓所包围的缺陷面积
# for i in range(len(contours)):
#     print(i)
#     # int
#     area = contourArea(contours[i])
#     print(area)

# 二值图判断
# for i in range(254):
#     if (i + 1) in result:
#         print("非二值图")
# print("二值图！")

# # 霍夫圆切割图片
# img = cv2.imread(path, 0)
# img = cv2.medianBlur(img, 5)
# cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=5000, param1=150, param2=50, minRadius=800)
# print('circles1:', circles)
# circles = np.uint16(np.around(circles))
# print('circles2:', circles)
#
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(cimg, (i[0], i[1]), i[2] - 80, (255, 255, 255), -1)
#     # draw the center of the circle
#     cv2.circle(cimg, (i[0], i[1]), 2, (255, 255, 255), 3)
# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.imshow("win", cimg)
# cv2.imwrite(r"..\..\source\opencv_pic\pre_mask.jpg", cimg)
# cv2.waitKey()
# cv2.destroyAllWindows()
