import cv2
import numpy as np
from PIL import Image

# # 读取
from matplotlib import pyplot as plt
#
# img = cv2.imread("sources/59.jpg")
# # cv2.imshow("img_1", img)
#
# # # BGR转RGB
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img = Image.fromarray(img)
# # img.show()
#
# # # 转灰度图
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("img_1", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 创建图片
# img_data = np.zeros((300, 300, 3), dtype=np.uint8)
# # print(img_data.shape)
# # 色彩通道默认是RGB，然而cv读取图片数据会以BGR的格式读取
# img_data[..., 0] = 255  # 变成蓝色
# # cv2.imshow("new", img_data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # # 视频读取
# # cap = cv2.VideoCapture("视频路径")
# # while True:
# #     ret, frame = cap.read()
# #     cv2.imshow("frame", frame)
# #     # 接收键盘输入q就退出循环
# #     # 一秒24帧，一帧大概就42ms
# #     if cv2.waitKey(42) & 0xFF == ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()
#
# # 绘图
# # cv2.line(img, (100, 100), (200, 100), color=(0, 0, 255), thickness=3)
# # cv2.circle(img, (50, 50), 20, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
# # thickness=-1表示图形填充
# # cv2.ellipse(img, (50, 50), (20, 30), 0, 0, 360, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# # cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2.putText(img, "what", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, lineType=cv2.LINE_AA)
# # cv2.imshow("line", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图片通道变换(通过阈值限制改变色彩通道的值)-RGB转灰度再转为二值图（非黑即白，一个通道）
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# # print(ret)
# # ret即阈值
# # 自适应阈值，局部像素点使用的局部的阈值，而不是只使用一个全局阈值
# # 针对亮度不同的图片可以使用
# # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 0)
# # cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图像的运算
# # 加减法，图像融合
# img1 = cv2.imread("sources/1.jpg")
# img2 = cv2.imread("sources/6.jpg")
# img = cv2.subtract(img1, img2)
# # 给两张图片加一个权重
# img = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
# # img = cv2.add(img1, img2)
# # cv2.imshow("add", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图像位运算-抠图
# # 前提：图片为二值图
# img1 = cv2.imread("sources/1.jpg")
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
# print(mask.shape)
# # 二值图取反
# # 非
# reverse_binary = cv2.bitwise_not(binary)
# # 与运算，需要先将值转为二进制再去做与运算，eg:11111111(255)和01010101
# # 与255运算，等于原值
# img_and = cv2.bitwise_and(img1, mask)
# # 或运算，与255运算，计算后就是255
# img_or = cv2.bitwise_or(img1, mask)
# # 异或运算
# # 和0异或，等于原值；和255异或会改变原值
# img_xor = cv2.bitwise_xor(img1, mask)
#
# # cv2.imshow("binary", img1)
# # cv2.imshow("rv_binary", reverse_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图像的几何变换（基于img1）
# # 目的是去噪
# h, w, c = img1.shape
# # resize
# dst = cv2.resize(img1, (w * 2, h * 2))
# # transpose
# dst = cv2.transpose(img1)
# # flip
# dst = cv2.flip(img1, 0)
#
# # cv2.imshow("binary", dst)
# # cv2.imshow("rv_binary", reverse_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图像的仿射变换（基于img1）????
# # 矩阵乘法，实现图像的缩放、旋转、平移、倾斜、镜像
# M = np.float32([[1, 0, 50], [0, 1, 50]])
# print(M.shape)
# # M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# # M = np.float32([[-0.5, 0, cols // 2], [0, 0.5, 0]])
# # M = np.float32([[1, 0.5, 0], [0, 1, 0]])
# # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.7)
# dst = cv2.warpAffine(img1, M, (w, h))
# print(dst.shape)
# # cv2.imshow("binary", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 透视变换，将图像拉伸铺满全屏
# img = cv2.imread("4.jpg")
# pts1 = np.float32([[25, 30], [179, 25], [12, 188], [189, 190]])
# pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img, M, (200, 201))
# cv2.imshow("src", img)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)

# 操作前，只能处理二值化图像
# 膨胀操作，让颜色值大的像素变得更粗
# img = cv2.imread("11.jpg", 0)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# dst = cv2.dilate(img, kernel)
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# 腐蚀操作，让颜色值大的像素变得更细
# img = cv2.imread("11.jpg", 0)
# kernel = cv2.getStructuringElement(cv.MORPH_RECT, (5, 5))
# dst = cv2.erode(img, kernel)
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 开操作，先腐蚀再膨胀（去白噪）

# img = cv2.imread("10.jpg", 0)
# kernel = cv2.getStructuringElement(cv.MORPH_RECT, (3, 3))
# dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
# cv2.imshow('src', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 闭操作：先膨胀再腐蚀（去除颜色值高的区域附近的黑色空洞）
# cv2.MORPH_CLOSE

# 两个都想去掉，那就先闭操作再做开操作

# 梯度操作 膨胀-腐蚀 轮廓
# 礼帽操作 原图-开  噪声
# 黑帽操作 闭-原图 空洞

# 滤波就是卷积操作，提取图片中的部分特征
# 高通滤波和低通滤波
# 低通滤波也叫平滑滤波，可以使图像变模糊，主要用于去噪
# 高通滤波一般用于获取图像边缘、轮廓或梯度
#
# # 均值滤波（模糊处理）
# src = cv2.imread("sources/5.jpg")
# # dst = cv2.blur(src, (5, 5))
# # # 高斯滤波
# # dst = cv2.GaussianBlur(src, (5, 5), 0)
# # 中值滤波（去除椒盐噪声，有白和黑的噪声）
# dst = cv2.medianBlur(src, 5)
# # # 双边滤波（连接本该连接上却断开的线条）
# # dst = cv2.bilateralFilter(src, 9, 75, 75)
# # # 高通滤波
# # dst = cv2.Laplacian(src, -1, ksize=3)
# cv2.imshow("src show", src)
# cv2.imshow("dst show", dst)
# cv2.waitKey(0)

# # 直方图均值化（去雾，让色域扩大）
# img = cv2.imread('sources/16.jpg')
# img_B = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.plot(img_B, label='B', color='b')
# img_G = cv2.calcHist([img], [1], None, [256], [0, 256])
# plt.plot(img_G, label='G', color='g')
# img_R = cv2.calcHist([img], [2], None, [256], [0, 256])
# plt.plot(img_R, label='R', color='r')
# plt.show()
# # 去雾效果
# img = cv2.imread('sources/16.jpg', 0)
# cv2.imshow("src", img)
# his = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.plot(his, label='his', color='r')
# # plt.show()
# dst = cv2.equalizeHist(img)
# cv2.imshow("dst", dst)
# cv2.waitKey()
# # cv2.imwrite("15.jpg", dst)
# his = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt.plot(his, label='his', color='b')
# plt.show()
#
# Canny算法
# 边缘提取算法
img = cv2.imread("sources/1.jpg", 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 20, 150)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
# 步骤
# 彩色图像转换为灰度图像
# 高斯滤波，滤除噪声点
# 计算图像梯度，根据梯度计算边缘幅值与角度
# 非极大值抑制[梯度和边缘都是高频信号，但是梯度线宽不是1，而边缘的线宽是一个像素]
# 双阈值边缘连接处理
# 二值化图像输出结果
#
# # 轮廓是一堆点，而边缘和梯度是线条
# # 前提二值化图像
# # cv2.findContours和cv2.drawContours
