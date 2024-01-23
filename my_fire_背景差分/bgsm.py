# -*- coding: utf-8 -*-
# import sys
# sys.path.append('/media/xl/021CA49C1CA48BEB/packages/opencv-3.4.1-with-contrib/build/lib')
import numpy as np
import cv2
from util import PointsRecorder
import time
from predict_image import predict_imgs


class FirePredictor(object):
    def __init__(self, kernel_size=3, perimeterThreshold=120, bg_threshold=80, record_len=10, valid_num=8):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.recorder = PointsRecorder(record_len=record_len, num_thr=valid_num, dist_thr=100)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.perimeter_thr = perimeterThreshold
        self.bg_threshold = bg_threshold

    def work(self, frame):
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        im, contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers_list = []
        for c in contours:
            # 计算各轮廓的周长
            perimeter = cv2.arcLength(c, True)
            if perimeter > self.perimeter_thr:
                # 找到一个直矩形（不会旋转）
                x, y, w, h = cv2.boundingRect(c)
                _cropImg = im[y:y + h, x:x + w]
                center = [x + w / 2, y + h / 2]

                if np.average(_cropImg) > self.bg_threshold:
                    if self.recorder.warning_flag(center):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    centers_list.append(center)
        self.recorder.update(centers_list)
        cv2.imshow('frame', frame)
        cv2.imshow('fgmask', fgmask)
        cv2.waitKey(1)


def check_worker():
    cap = cv2.VideoCapture('/home/room/PycharmProjects/my_fire_bgsm/output3.avi')
    P = FirePredictor()
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        index += 1
        if index % 5 == 0:
            P.work(frame)


def bgsm_show():
    R = PointsRecorder()
    # BackgroundSubtractorMOG2
    # opencv自带的一个视频
    cap = cv2.VideoCapture('/home/room/PycharmProjects/my_fire_bgsm/output3.avi')
    # cap = cv2.VideoCapture('/home/lijun/Videos/fire.mkv')
    # 创建一个3*3的椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 创建BackgroundSubtractorMOG2
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    # fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
    count = 0
    fps = int(cap.get(5))
    size = (int(cap.get(3)), int(cap.get(4)))
    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # save_video = cv2.VideoWriter('output.avi',fourcc,fps,size)

    while (1):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 5 != 0:
            continue

        start = time.time()
        fgmask = fgbg.apply(frame)
        # print('It costs %f ms.' % ((time.time()-start)*1000))
        # 形态学开运算去噪点
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # 寻找视频中的轮廓
        im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        per_list = []
        centers_list = []
        for c in contours:
            # 计算各轮廓的周长
            perimeter = cv2.arcLength(c, True)
            per_list.append(perimeter)
            print("perimeter: ", per_list)
            if (perimeter > 100) and (perimeter < 500):
                # 找到一个直矩形（不会旋转）
                x, y, w, h = cv2.boundingRect(c)
                _cropImg = im[y:y + h, x:x + w]
                center = [x + w / 2, y + h / 2]

                # Img = frame[y:y + h, x:x + w, :]
                # cv2.imshow("_cropImg: ", Img)
                # cv2.waitKey(0)

                """火焰检测模型（128.128）"""
                threshold_fire = predict_imgs(frame[y:y + h, x:x + w, :])
                # if np.average(_cropImg) > 80 and threshold_fire == 1:
                # #if np.average(_cropImg) > 80:
                #     if R.warning_flag(center):
                #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #         print('Fire on -------------------------threshold_fire: %d', threshold_fire)
                #         # time.sleep(15)
                #     centers_list.append(center)
                if np.average(_cropImg) > 80:
                    if threshold_fire == 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print('Fire on -------------------------threshold_fire: %d', threshold_fire)

        # save_video.write(frame)   #对火焰的存储
        R.update(centers_list)
        cv2.imshow('frame', frame)
        cv2.imshow('fgmask', fgmask)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cap.release()
    # save_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # R = Recorder()
    # print(R.dist_thr)
    bgsm_show()
    check_worker()
