#!/usr/bin/python
# coding:utf8

# 主要参考代码
import cv2
import numpy as np
from predict_image import predict_imgs

# camera = cv2.VideoCapture("/home/lijun/Videos/firemanyface_mv.avi")
camera = cv2.VideoCapture(
    "/home/room/PycharmProjects/my_fire_bgsm/output3.avi")
# camera = cv2.VideoCapture(0)  # 参数0表示第一个摄像头
# camera = cv2.VideoCapture("/home/lijun/workspace/火焰检测/1.mp4")
if (camera.isOpened()):  # 判断视频是否打开
    print('Open')
else:
    print('摄像头未打开')

# 构建椭圆结果
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((3, 3), np.uint8)
background = None
background_bak = None
count = 0
tag = True

fps = int(camera.get(5))
size = (int(camera.get(3)), int(camera.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'MP42')
save_video = cv2.VideoWriter('output3.avi', fourcc, fps, size)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # 读取视频流
    ok, frame_lwpCV = camera.read()
    if not ok:
        camera.release()
        break
    count += 1
    if count % 1 != 0:
        continue

    # 对帧进行预处理，>>转灰度图>>高斯滤波（降噪：摄像头震动、光照变化）。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    diff = fgbg.apply(gray_lwpCV)

    per_list = []
    # 显示矩形框：计算一幅图像中目标的轮廓
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:

        perimeter = cv2.arcLength(c, True)
        per_list.append(perimeter)
        max_id = np.argmax(per_list)
        p3 = (100, 100)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # title = str(per_list[max_id])
        # cv2.putText(frame_lwpCV, title, p3, font, 3, (0, 0, 255), 2)
        print("********max: ", per_list[max_id])

        # if (perimeter > 120) and (perimeter < 1000):
        if perimeter > 120:
            (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
            _cropImg = frame_lwpCV[y:y + h, x:x + w, :]
            fire_threshold = predict_imgs(_cropImg)
            if fire_threshold == 0:
                cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

    save_video.write(frame_lwpCV)  # 对火焰的存储
    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('dis', diff)
    cv2.waitKey(1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按'q'健退出循环
        break

save_video.release()
# 释放资源并关闭窗口
camera.release()
# cv2.destroyAllWindows()
