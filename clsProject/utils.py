import os
import random

data_path = r"..\..\source\enzyme\data"
train_file = r"..\..\source\enzyme\train\train_file.txt"
# train_file = r"train"
test_file = r"..\..\source\enzyme\test\test_file.txt"
train_0_file = r"..\..\source\enzyme\train\train_0_file.txt"


def split_data():
    lens = []
    # w模式能在多级目录下创建文件，可以用os.makedirs()
    # 前提是你的所有目录都存在，只是文件不存在
    train = open(train_file, 'w')
    test = open(test_file, 'w')
    #  将原始数据整理为数据加上标签的形式
    for i, file in enumerate(os.listdir(data_path)):
        file_path = os.path.join(data_path, file)
        with open(file_path) as f:
            # 记录写入的行数
            count = 0
            # 记录测试集写入的行数
            test_count = 106 if i == 0 else 27
            for line in f:
                line.split()
                # print(len(line))
                # 去除空行
                if len(line) != 1:
                    # 去除数字行的换行符
                    data_file = test if count > test_count else train
                    if not line[0].isdigit():
                        # print(line)
                        length = len(line[:-1])
                        lens.append(length)
                        # 在换行符之前拼接标签元素，并换行
                        line = line[:-1] + '.' + file[0] + '\n'
                        data_file.write(line)
                        count += 1
    train.close()
    test.close()
    return max(lens), min(lens)


# 将训练集进行增样以及resize操作，对测试集进行resize操作（加0来统一数据的长度）
# 训练集处理


def sample_add(l0, l1, max_len, min_len, is_train=True):
    if is_train:
        r_list0 = []
        r_list1 = []
        # 扩充负样本
        for _ in range(1200):
            x, y = [random.randint(0, len(l0) - 1) for _ in range(2)]
            x1, y1 = [random.randint(0, len(l0[x]) - 1) for _ in range(2)]
            # print(min_len // 3)
            x2, y2 = [random.randint(0, len(l0[y]) - 1) for _ in range(2)]
            res_str = l0[x][x1:y1] + l0[y][x2:y2]
            # print(len(l0[x]),len(l0[y]))
            # print(x1, y1, x2, y2)
            if min_len <= len(res_str) <= max_len:
                r_list0.append(res_str)
        # 扩充正样本
        for _ in range(500):
            x, y = [random.randint(0, len(l1) - 1) for _ in range(2)]
            x1, y1 = [random.randint(0, len(l1[x]) - 1) for _ in range(2)]
            # print(min_len // 3)
            x2, y2 = [random.randint(0, len(l1[y]) - 1) for _ in range(2)]
            res_str = l1[x][x1:y1] + l1[y][x2:y2]
            if min_len <= len(res_str) <= max_len:
                r_list1.append(res_str)
        return r_list0, r_list1


if __name__ == '__main__':
    # 划分训练集和测试集
    # print(split_data())
    max_len, min_len = split_data()
    # 训练集增样
    list0 = []
    list1 = []
    with open(train_file) as t:
        for line in t:
            # 一行末尾的换行符一定要去除，不一定会显示
            flag = line.split('.')[1].strip('\n')
            if flag == "0":
                list0.append(line.split('.')[0])
            else:
                list1.append(line.split('.')[0])
    neg_file = open(train_0_file, 'w')
    r_list0, r_list1 = sample_add(list0, list1, max_len, min_len, True)[0:2]
    print(len(r_list0), len(r_list1))
    # print(len(list0) + len(list1))
    for i in r_list0:
        neg_file.write(i + '.0' + '\n')
    for i in r_list1:
        neg_file.write(i + '.1' '\n')
    neg_file.close()
