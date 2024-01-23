# -*- coding: utf-8 -*-

class AlgorithmBase(object):
    def __init__(self, min_FPS, score_threshold, InputQ, OutputQ, img_Length):
        self.MinFPS = min_FPS
        self.ScoreThreshold = score_threshold

        self.InputQ = InputQ
        self.OutputQ = OutputQ
        self.Length = img_Length

    def start(self):
        """
        Initialize algorithmk
        :return:
        """

        raise NotImplementedError()

    def kill(self):
        """
        Kill working thread
        """

        raise NotImplementedError()
