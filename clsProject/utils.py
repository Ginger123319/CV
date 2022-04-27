import os
import random

import numpy as np
import torch

data_path = r"..\..\source\enzyme\data"
train_file = r"..\..\source\enzyme\train\train_file.txt"
test_file = r"..\..\source\enzyme\test\test_file.txt"


# train = open(train_file, 'w')
# test = open(test_file, 'w')
# #  将原始数据整理为数据加上标签的形式
# for file in os.listdir(data_path):
#     print(file)
#     file_path = os.path.join(data_path, file)
#     with open(file_path) as f:
#         # strs_list = f.readlines()
#         # print(strs_list)
#         # 记录行数
#         count = 0
#         for line in f:
#             line.split()
#             # print(len(line))
#             # 去除空行
#             if len(line) != 1:
#                 # 去除数字行的换行符
#                 data_file = test if count < 8 else train
#                 if line[0].isdigit():
#                     # data_file.write(line[:-1])
#                     # print(line[:-1])
#                     # count += 1
#                     print(line)
#                 else:
#                     # 在换行符之前拼接标签元素，并换行
#                     line = line[:-1] + '.' + file[0] + '\n'
#                     # print(line)
#                     data_file.write(line)
#                     count += 1
# 将训练集进行增样以及resize操作，对测试集进行resize操作（加0来统一数据的长度）
# 训练集处理

def sample_add(l0, l1, max_len, min_len, is_train=True):
    if is_train:
        # print(len(l0))
        for i, elem in enumerate(l0):
            l0[i] = elem.ljust(max_len, "0")
        for i, elem in enumerate(l1):
            l1[i] = elem.ljust(max_len, "0")
        # print(len(l0[30]))
        # print(l1)
        # print(target_len)
        r_list0 = []
        r_list1 = []
        # print(len(r_list1))
        for i in range(300):
            x, y = [random.randint(0, len(l0) - 1) for _ in range(2)]
            x1, y1 = [random.randint(0, min_len // 8) for _ in range(2)]
            # print(min_len // 3)
            x2, y2 = [random.randint(min_len // 3, min_len - 1) for _ in range(2)]
            res_str = l0[x][x1:x2] + l0[y][y1:y2]
            # print(len(l0[x]),len(l0[y]))
            # print(x1, y1, x2, y2)
            if min_len <= len(res_str) <= max_len:
                r_list0.append(res_str)

        for i in range(100):
            x, y = [random.randint(0, len(l1) - 1) for _ in range(2)]
            x1, y1 = [random.randint(0, min_len // 8) for _ in range(2)]
            # print(min_len // 3)
            x2, y2 = [random.randint(min_len // 3, min_len - 1) for _ in range(2)]
            res_str = l0[x][x1:x2] + l0[y][y1:y2]
            if min_len <= len(res_str) <= max_len:
                r_list1.append(res_str)
        # print(res_str)
        # print(len(res_str))
        # print(x1, y1, x2, y2)

        for i, elem in enumerate(r_list0):
            r_list0[i] = elem.ljust(max_len, "0")
        for i, elem in enumerate(r_list1):
            r_list1[i] = elem.ljust(max_len, "0")
        r_list0.extend(l0)
        r_list1.extend(l1)
        # print(len(r_list0), len(r_list1))

        # s = "LPTSNPAQELEARQLGR".ljust(32,'0')
        # print(s)
        # print(len(r_list0[i]))
        # print(r_list0[i])
        # exit()
        # print(len(l0[5]), len(l1[5]))

        # print(len(r_list0[100]), len(r_list1[100]))
        return r_list0, r_list1, l0, l1
    else:
        for i, elem in enumerate(l0):
            l0[i] = elem.ljust(632, "0")
        for i, elem in enumerate(l1):
            l1[i] = elem.ljust(632, "0")
        # print(len(l0[2]))
        return l0, l1


lens = []
list0 = []
list1 = []
with open(test_file) as t:
    # data = t.readlines()
    # print(data)
    for line in t:
        # print(line.split('.')[0])
        length = len(line.split('.')[0])
        lens.append(length)
        # print(line.split('.')[1])
        # 一行末尾的换行符一定要去除，不一定会显示
        flag = line.split('.')[1].strip('\n')
        # print(flag)
        if flag == "0":
            list0.append(line.split('.')[0])
        else:
            list1.append(line.split('.')[0])
    # print(max(lens))
    # print(len(list0))
    # print(len(list1))
if __name__ == '__main__':
    sample_add(list0, list1, max(lens), min(lens), False)
    # print(len(list0) + len(list1))
