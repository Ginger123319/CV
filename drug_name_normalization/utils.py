import cfg
import numpy as np


class Utils:
    def __init__(self):
        self.cls_list = list()
        self.input_max_len = 0
        self.output_max_len = 0
        with open(cfg.save_path, encoding="utf-8") as f:
            f.readline()
            for line in f.readlines():
                line = line.split()
                if line[1] not in self.cls_list:
                    self.cls_list.append(line[1])
                if len(line[0]) > self.input_max_len:
                    self.input_max_len = len(line[0])
                if len(line[1]) > self.output_max_len:
                    self.output_max_len = len(line[1])
        # print(len(self.cls_list))
        # print(self.cls_list)
        # print(self.input_max_len)
        # print(self.output_max_len)

    def get_max_len(self):
        return self.input_max_len, self.output_max_len

    def get_name_list(self):
        return self.cls_list


if __name__ == '__main__':
    utils = Utils()
    print(utils.get_max_len())
    print(utils.get_name_list())
