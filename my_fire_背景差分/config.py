# -*- coding: utf-8 -*-

DEBUG = True

GLOBAL_MIN_FPS = 5

function_types_list = ['boundary', 'fire', 'leakage']
Q_len = 100
draw_line_width = 3
image_save_resize_ratio = .3
min_fps = 1


class FireConstants(object):
    def __init__(self):
        self.kernel_size = 3
        self.perimeterThreshold = 120
        self.bg_threshold = 80
        self.record_len = 10
        self.valid_num = 8
        self.area_poly = []
