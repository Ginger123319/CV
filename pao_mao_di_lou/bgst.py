# 背景差分demo
import cv2
import math

import torch.cuda

# cv2.imread()	numpy.ndarray	BGR
# cv2.VideoCapture()	numpy.ndarray	BGR
# PIL（Python Image Library）	PIL.Image.Image	RGB
# 提取轮廓，计算面积
print(torch.cuda.is_available())


def bgcf_knn(path):
    cap = cv2.VideoCapture(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorKNN()
    cnt = 0
    if cap.isOpened():
        fps = cap.get(5)
        print("fps is ", fps)

        frameNum = int(cap.get(7))
        print("frame num is ", frameNum)
        while 1:
            success, frame = cap.read()
            # print(frame)
            if success:
                frame = frame[5:375, 1055:1370]
                cnt += 1
                fgmask = fgbg.apply(frame)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                if cnt % 5 == 0:
                    print("cnt", cnt)
                    cv2.imshow('frame', fgmask)
                    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    # cv2.imshow('frame', frame)
                    interval = cv2.waitKey(40)
                    if interval & 0xff == ord('q'):
                        break
            else:
                print("读取视频结束，释放资源")
                cv2.destroyAllWindows()
                break
    cap.release()
    cv2.destroyAllWindows()


def bgcf(path):
    cnt = 0
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = cap.get(5)
        print(fps)

        frameNum = int(cap.get(7))
        print(frameNum)
        while True:
            success, frame = cap.read()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(frame.shape)
                # cv2.imshow('frame', frame)
                # interval = cv2.waitKey(20)
                # if interval & 0xff == ord('q'):
                #     break
                if cnt == 0:
                    background = cv2.copyTo(frame, frame)
                    # cv2.imshow('background', background)
                    interval = cv2.waitKey(20)
                    if interval & 0xff == ord('q'):
                        break
                else:
                    # 第二帧开始做差分
                    # 背景和原图相减
                    subpic = abs(frame - background)
                    print("sub.shape", subpic)
                    # 对差分结果进行二值化处理
                    _, result = cv2.threshold(subpic, 100, 255, cv2.THRESH_BINARY)
                    print("result", result.shape)
                    # cv2.imshow("sub", subpic)
                    cv2.imshow("result", result)
                    # cv2.imshow("frame", frame)
                    interval = cv2.waitKey(20)
                    if interval & 0xff == ord('q'):
                        break
                    background = frame
                print(cnt)
                cnt += 1

            else:
                print("读取视频结束，释放资源")
                cv2.destroyAllWindows()
                break
    cap.release()


if __name__ == '__main__':
    bgcf_knn("/home/room/视频/炼化/smoke_short.mp4")
