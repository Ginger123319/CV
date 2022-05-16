import os
import cfg
import csv
import random
from sklearn import preprocessing
import pandas as pd

# _open, _high, _low, _close, _volume, _amount, _turn, _pctChg
file_namelist = os.listdir(cfg.train_dir)
# 读取csv数据，根据需求选择遍历train_dir或者test_dir
# 需要手动调控
for filename in file_namelist:
    filepath = os.path.join(cfg.train_dir, filename)
    # print(filename)
    with open(filepath) as f:
        reader = pd.read_csv(f)
        print(reader)
        # 跳过首行表头
        # next(reader)

        # df_z_score = pd.DataFrame(z_score, index=reader.index, columns=reader.columns)
        # df_z_score
