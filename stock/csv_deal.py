import os
import cfg
import csv
import random

# _open, _high, _low, _close, _volume, _amount, _turn, _pctChg
file_namelist = os.listdir(cfg.train_dir)
# 读取csv数据，根据需求选择遍历train_dir或者test_dir
# 需要手动调控
for filename in file_namelist:
    filepath = os.path.join(cfg.train_dir, filename)
    # print(filename)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        # 跳过首行表头
        next(reader)
        table_list = []
        # 遍历表中的每一行
        for row in reader:
            row_list = [row["open"], row["high"], row["low"], row["close"], row["volume"], row["turn"], row["pctChg"]]
            if "" in row_list:
                continue
            row_list = list(map(float, row_list))
            table_list.append(row_list)
        # 表中数据少于180条的数据丢弃
        if len(table_list) < 180:
            continue
        # 生成30个不同的随机数字作为随机索引
        rand_list = [i for i in range(len(table_list) - 6)]
        rand_list = random.sample(rand_list, 30)
        # 根据索引，取出五个样本数据，以及标签
        for index in rand_list:
            _elem = [elem[0:-1] for elem in table_list[index:index + 5]]
            _tag = 0 if table_list[index + 5][-1] <= 0 else 1
            # 生成测试数据使用
            # if _tag == 0:
            #     continue
            # 生成训练数据需要注释上面两行代码
            # 将每一个样本数据和标签并作一行写入txt文件中
            # 根据需求将数据写入test_file或者data_file
            with open(cfg.train_file, 'a') as fi:
                fi.write(str(_tag) + ":" + str(_elem))
                fi.write("\n")
print("success")
