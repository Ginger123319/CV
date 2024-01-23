# -*- coding: utf-8 -*-
# import sys
# sys.path.append('/home/xl/inst/mxnet-1.5.0/python')
import mxnet
import numpy as np
import cv2
import os
from Queue import Queue
import logging
import threading
import time
import config as cfg
from config import FireConstants as FireC
import logging
from util import PointsRecorder, transfer_pts_to_int, get_dict_key, draw_poly
from AlgorithmBase import AlgorithmBase
import sys

# print(sys._getframe().f_lineno)

"""对火焰检测的具体算法"""


class FirePredictor(object):
    def __init__(self, kernel_size=3, perimeterThreshold=120, bg_threshold=80, record_len=10, valid_num=8,
                 area_poly=[]):
        """函数会返回指定形状和尺寸的结构元素"""
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.recorder = PointsRecorder(record_len=record_len, num_thr=valid_num, dist_thr=100)
        """
        BackgroundSubtractorMOG2用于动态目标检测，用到的是基于自适应混合高斯背景建模的背景减除法，
        相对于BackgroundSubtractorMOG，其具有更好的抗干扰能力，特别是光照变化
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.perimeter_thr = perimeterThreshold
        self.bg_threshold = bg_threshold
        self.area_poly = area_poly

    """
    fgmask	=	cv.BackgroundSubtractorMOG2.apply(	image[, fgmask[, learningRate]]	)
    Parameters
    image	Next video frame. Floating point frame will be used without scaling and should be in range [0,255].
    fgmask	The output foreground mask as an 8-bit binary image.
    learningRate	The value between 0 and 1 that indicates how fast the background model is learnt. 
                    Negative parameter value makes the algorithm to use some automatically chosen learning rate. 
                    0 means that the background model is not updated at all, 
                    1 means that the background model is completely reinitialized from the last frame.
    MorphologyEx()进行更多的形态学变换
    findContours()寻找图像轮廓
    arcLength()计算轮廓的周长
    contourArea和arcLength检测物体的轮廓面积和周长
    Bounding Rectangle()矩形边框,是说，用一个最小的矩形，把找到的形状包起来
    double pointPolygonTest(InputArray contour, Point2f pt, bool measureDist)功能可查找图像中的点与轮廓之间的最短距离. 当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零.
    contour – 输入findContour提取到的边缘.
    pt – 需要检测的点.
    measureDist – 为真，则计算检测点到边缘的距离，为负值在外部，0在边上，正值在内部。为假，则返回-1（在contour外部）、0（在contour上）、1（在contouri内部。
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行
    """

    def work(self, frame):
        bboxes = []
        Warning = False
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        im, contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers_list = []
        for c in contours:
            centered_in_poly = (self.area_poly == [])
            perimeter = cv2.arcLength(c, True)
            if perimeter > self.perimeter_thr:
                x, y, w, h = cv2.boundingRect(c)
                center = [x + w / 2, y + h / 2]
                _center = (x + w / 2, y + h / 2)
                """
                为检测目标划分了识别区
                """
                for poly in self.area_poly:
                    if cv2.pointPolygonTest(np.array(poly), _center, False) > 0:
                        centered_in_poly = True

                _cropImg = im[y:y + h, x:x + w]
                if np.average(_cropImg) > self.bg_threshold and centered_in_poly:
                    if self.recorder.warning_flag(center):
                        bboxes.append([x, y, x + w, y + h])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        Warning = True
                    centers_list.append(center)
        if Warning:
            for poly in self.area_poly:
                draw_poly(poly, frame)
        self.recorder.update(centers_list)
        # cv2.imshow('frame', frame)
        # cv2.imshow('fgmask', fgmask)
        # cv2.waitKey(1)
        return Warning, frame, bboxes


"""总控制器"""


class FirePredictorController(AlgorithmBase):
    def __init__(self, InputQ, OutputQ, min_FPS=5, score_threshold=.4, img_Length=500.0,
                 perimeterThreshold=120, bg_threshold=80, record_len=10, valid_num=8, kernel_size=3):
        AlgorithmBase.__init__(self, min_FPS, score_threshold, InputQ, OutputQ, img_Length)
        '''default parameters'''
        self.perimeterThreshold = perimeterThreshold
        self.bg_threshold = bg_threshold
        self.record_len = record_len
        self.valid_num = valid_num
        self.kernel_size = kernel_size

        '''线程列表'''
        self.threads_list = []
        self._InfoDict = {}  # 信息字典
        self.running_flag = False  # 系统是否运行中
        # self.OutputQ = OutputQ

    '''更新相机信息'''

    def update_camera_info(self, camera_info_dict, task='on'):

        '''
        :param camera_info_dict:
        {
            camera_id:{'fps': ###,
                        'size': (W,H),
                        'perimeterThreshold':120,
                        'bg_threshold':80,
                        'record_len':10,
                        'valid_num':8,
                        'kernel_size':3,
                        'area_poly':[[(x0,y0),(x1,y1),...,(xn,yn),(x0,y0)],
                                        [(x0,y0),(x1,y1),...,(xn,yn),(x0,y0)]] or []
                        ...
            }
        }
        if True, we add a camera with initialization, else, delete a camera.
        camera_id: string
        task:string 'on' or 'off'
        '''

        '''
        assert 判断task是否是"on" 或者 "off", 如果不是则报错 检查条件，不符合就终止程序
        '''
        assert task in ['on', 'off']
        if self.running_flag:
            self.kill()

        '''如果相机在列表里，则为此相机的算法配置参数'''
        for cameraID, info in camera_info_dict.items():
            C = FireC()
            if task == 'on':
                '''New camera opened!'''
                # newCamera = (cameraID not in self._InfoDict)
                self._InfoDict[cameraID] = info
                perimeterThreshold = get_dict_key(info, 'perimeterThreshold', C.perimeterThreshold)
                bg_threshold = get_dict_key(info, 'bg_threshold', C.bg_threshold)
                record_len = get_dict_key(info, 'record_len', C.record_len)
                valid_num = get_dict_key(info, 'valid_num', C.valid_num)
                kernel_size = get_dict_key(info, 'kernel_size', C.kernel_size)
                _area_poly = get_dict_key(info, 'area_poly', C.area_poly)
                area_poly = transfer_pts_to_int(_area_poly)

                '''给相机配置好参数后启动算法'''
                self._InfoDict[cameraID]['worker'] = FirePredictor(kernel_size, perimeterThreshold, bg_threshold,
                                                                   record_len, valid_num, area_poly)
                '''获取用于预处理的信息'''
                if self._InfoDict[cameraID]['fps'] > self.MinFPS:
                    self._InfoDict[cameraID]['skip_interval'] = int(self._InfoDict[cameraID]['fps'] / self.MinFPS)
                else:
                    self._InfoDict[cameraID]['skip_interval'] = 1
                self._InfoDict[cameraID]['resize_ratio'] = float(self.Length / max(self._InfoDict[cameraID]['size']))

                '''数据转换成 int'''
                self._InfoDict[cameraID]['area_poly'] = area_poly

            elif task == 'off' and cameraID in self._InfoDict:
                # 删除一个相机
                del (self._InfoDict[cameraID])
        '''如果相机的信息字典不为空，则开始运行'''
        if self._InfoDict != {}:
            self.start()

    '''线程函数： 如果此线程的运行指令为真，则一直进行检测'''

    def loop(self):
        # index = 0
        while self.running_flag:
            '''从相机的队列中获取相机的ID和相机的frame, 再从frame中每秒取5帧进行计算'''
            frame, frameID, cameraID = self.InputQ.get()
            if cameraID in self._InfoDict:
                skip_interval = self._InfoDict[cameraID]['skip_interval']
                # print('Preprocess %d.' % frameID)
                if frameID % skip_interval == 0:
                    ''''将启动算法的实例赋给worker, 如果有报警则返回给ret'''
                    worker = self._InfoDict[cameraID]['worker']
                    ret, frame, bboxes = worker.work(frame)
                    if ret:
                        '''输出队列是报警的一些信息'''
                        self.OutputQ.put((frame, frameID, cameraID, bboxes))
                    # index = (index + 1) % self.NUM_thread
            else:
                logging.info('Warning: CameraID %s doesnot existed!' % cameraID)

    '''每启动一次就启动一个线程，'''

    def start(self):
        self.running_flag = True
        T = threading.Thread(target=self.loop, args=())
        '''将主线程设置为守护进程，只要主线程结束了，不管子线程是否完成，一并和主线程退出'''
        T.setDaemon(True)
        T.start()
        '''线程列表,将线程加入线程列表'''
        self.threads_list.append(T)

    '''
    非守护进程join不影响，加上join会强制让守护进程在主线程之前结束，
    加参数是等一定时间守护线程还不结束的话，主线程就要强制关闭了 
    '''

    def kill(self):
        self.running_flag = False
        time.sleep(0.1)  #
        states_list = [t.is_alive() for t in self.threads_list]
        logging.info('All states for workers after killing, %s.' % str(states_list))
        try:
            for t in self.threads_list:
                t.join(1)
        except:
            logging.warning('Join threads failed.')
            logging.info('All states for workers after killing, %s.' % str(states_list))
            # pass


'''输入的信息格式'''
'''
    camera_info_dict:
    {
        camera_id:{'fps': ###,
                    'size': (W,H),
                    ...
        }
    }
    dataitem from inputQ:
    [frame,frameID,cameraID]
'''

'''清空相机的队列'''


def clear_queue(Q):
    index = 0
    start = time.time()
    while True:
        # time.sleep(0.05)
        frame, frameID, cameraID, bboxes = Q.get()
        # if frameID % 10 == 0:
        cv2.imwrite('DEBUG/%03d.jpg' % (frameID), frame)
        print("clear_queue: %d" % index)

        index += 1
        # print('It costs %f ms for each item.' % ((time.time()-start)*1000/index))


def check_all():
    print('mxnet version:%s' % mxnet.__version__)
    video = '/home/lijun/Videos/firemanyface_mv.avi'
    camera = cv2.VideoCapture(video)
    fps = int(camera.get(5))
    size = (int(camera.get(3)), int(camera.get(4)))

    inputQ, outputQ = Queue(20), Queue(20)
    P = FirePredictorController(inputQ, outputQ, 5)

    '''测试中需要清空输出的队列'''
    T = threading.Thread(target=clear_queue, args=(outputQ,))
    T.setDaemon(True)
    T.start()

    # T = threading.Thread(target=clear_queue, args=(P._InputQList[1],))
    # T.setDaemon(True)
    # T.start()

    camera_info_dict = {
        '0001': {
            'fps': fps,
            'size': size,
            'area_poly': [[(10, 10), (10, 1070), (500, 1070), (500, 10), (10, 10)]]
        },
        '0002': {
            'fps': fps,
            'size': size,
            'area_poly': [[(500, 10), (500, 1070), (1000, 1070), (1000, 10), (500, 10)],
                          [(1000, 10), (1000, 1070), (1500, 1070), (1500, 10), (1000, 10)]]
        },
    }
    P.update_camera_info(camera_info_dict)
    P.start()
    index = 0
    while True:
        ok, img = camera.read()
        if not ok:
            camera.release()
            P.kill()
            break
        start = time.time()
        inputQ.put((img, index, '0001'))
        end = time.time()
        # print('It costs %f ms.' % ((end - start) * 1000))
        start = end
        inputQ.put((img, index, '0002'))
        end = time.time()
        # print('It costs %f ms.' % ((end - start) * 1000))
        # start = end
        # inputQ.put((img, index, '0003'))
        # end = time.time()
        # print('It costs %f ms.' % ((end - start) * 1000))
        index += 1


if __name__ == '__main__':
    check_all()
    # import numpy as np
    # _pt = (100,100)
    # points = [(37,25),(400,25),(37,400),(37,25)]
    # print(cv2.pointPolygonTest(np.array(points),_pt,False))
    # video = '/home/xl/data/2018-12-02 15-25-46.mp4'
    # camera = cv2.VideoCapture(video)
    # # P=PersonPredict()
    # while True:
    #     ok,img = camera.read()
    #     if not ok:
    #         break
    #     for idx,pt in enumerate(points):
    #         if idx<len(points)-1:
    #             cv2.line(img,pt,points[idx+1],(255,0,0),3)
    #     cv2.imshow('img',img)
    #     cv2.waitKey(1)
    # bboxes = P.detect_interval(img)
