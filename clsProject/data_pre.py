import os
import numpy as np
import torch
# 非常宝贵的一次南辕北辙，搞清楚问题在行动吧
data_path = r"G:\liewei\source\enzyme\data"
train_file = r"G:\liewei\source\enzyme\train\train_file.txt"
test_file = r"G:\liewei\source\enzyme\test\test_file.txt"
train = open(train_file, 'w')
test = open(test_file, 'w')

for file in os.listdir(data_path):
    print(file)
    file_path = os.path.join(data_path, file)
    with open(file_path) as f:
        # strs_list = f.readlines()
        # print(strs_list)
        # 记录行数
        count = 0
        for line in f:
            line.split()
            # print(len(line))
            # 去除空行
            if len(line) != 1:
                # 去除数字行的换行符
                data_file = test if count // 2 < 8 else train
                if line[0].isdigit():
                    data_file.write(line[:-1])
                    print(line[:-1])
                    count += 1
                else:
                    # 在换行符之前拼接标签元素，并换行
                    line = line[:-1] + '.' + file[0] + '\n'
                    print(line)
                    data_file.write(line)
                    count += 1
