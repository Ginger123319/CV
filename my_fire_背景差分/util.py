# -*- coding: utf-8 -*-
import numpy as np
import cv2
import config as cfg

"""循环列表类"""


class CycleList(object):
    def __init__(self, length):
        self._len = length
        self._pointer = 0
        self._container = []
        self._full = False

    """
    在循环链表中插入新的数据，替换掉原来的旧数据
    """

    def insert(self, item):
        if not self._full:
            self._container.append(item)
            if len(self._container) >= self._len:
                self._full = True
        else:
            self._container[self._pointer] = item
        """这句是最关键的一句"""
        self._pointer = (self._pointer + 1) % self._len

    def get_len(self):
        return self._len

    def delete_last_item(self):
        del (self._container[self._pointer])
        self._full = False
        if self._pointer < 0:
            self._pointer = len(self._container) - 1

    """
    获得新的一个元组，存放有用数据  
    """

    def get_numpy_points(self):
        '''
        Assert that a item is [] or [(x0,y0),(x1,y1),...]
        :return: numpy array as array([[x0,y0],[x1,y1],...,[xn,yn]])
        '''
        points_list = []
        for points in self._container:
            for point in points:
                points_list.append(point)
        return np.array(points_list)


"""
对重要数据点的记录
"""


class PointsRecorder(object):
    def __init__(self, record_len=30, num_thr=25, dist_thr=100):
        self.CycleList = CycleList(record_len)
        self.num_thr = num_thr
        self.dist_thr = dist_thr

    def update(self, centerPoints):
        self.CycleList.insert(centerPoints)

    def warning_flag(self, centerPoint):
        # warning = False
        np_record = self.CycleList.get_numpy_points()
        if len(np_record) == 0:
            return False
        """中位数"""
        mid_pt = np.median(np_record, axis=0)
        centerPoint = np.array(centerPoint)

        """方差, 最接近中心的位置"""
        if np.sqrt(np.sum((mid_pt - centerPoint) ** 2)) > self.dist_thr:
            return False
        count = 0
        for point in np_record:
            if np.sqrt(np.sum((mid_pt - point) ** 2)) < self.dist_thr:
                count += 1
        print("PointsRecorder count: %d" % count)
        return (count > self.num_thr)


"""将图形点转换为一组整数元组，为了后面进行计算"""


def transfer_pts_to_int(points_list):
    """先申请一个元组变量，在讲数据加入里边，数据来源为points_list"""
    dist_poly_points = []
    for poly in points_list:
        dist_poly = []
        for point in poly:
            x, y = int(point[0]), int(point[1])
            dist_poly.append((x, y))
        dist_poly_points.append(dist_poly)
    return dist_poly_points


"""此函数主要为了划线"""


def draw_poly(points, img):
    for idx, pt in enumerate(points):
        if idx < len(points) - 1:
            """img 为在什么图上划线， pt为起始点， points为终止点， （255,0,0）为图形颜色， cfg.draw_line_width为线的粗细"""
            cv2.line(img, pt, points[idx + 1], (255, 0, 0), cfg.draw_line_width)


"""此函数为获得字典中关键字的直， 如果有关键字，就赋已有的直，没有关键字,就给出新的直"""


def get_dict_key(dict, key, default=None):
    if key in dict:
        return dict[key]
    else:
        return default


def check():
    C = CycleList(10)
    C.insert([(1, 4)])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([])
    C.insert([(3, 3)])
    C.insert([])
    C.insert([(2, 7)])
    C.insert([])
    C.insert([(4, 0)])
    C.insert([])
    C.insert([(5, 0)])
    print("check: np.sqrt: %d" % np.sqrt(np.sum(np.median(C.get_numpy_points(), axis=0) ** 2)))


def check2():
    D = {'a': 1}
    print("115", get_dict_key(D, 'a'))
    print("116", get_dict_key(D, 'a', 2))
    print("117", get_dict_key(D, 'b', 2))
    print("118", get_dict_key(D, 'b'))


if __name__ == '__main__':
    # check()
    check2()
