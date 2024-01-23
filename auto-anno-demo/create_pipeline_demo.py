#!/usr/bin/env python
# -*- coding: utf-8 -*-

from demo.my_model import MyModel
import pandas as pd
import main_utils
from pathlib import Path

# ===============================================
# 手动输入参数
input_dir = "exp_dir/input"  # 里面需要有文件：input_label.csv，input_added_label.csv，input_unlabel.csv
output_dir = "exp_dir/output"  # 件保存路径
is_first_train = True
query_cnt = 100
val_size = 0.2
strategy = "LeastConfidence"
options = {"epochs": 1,
           "batch_size": 16,
           "lr": 0.1,
           "lrf": 0.05}
model_id = None

# ===============================================
# 准备变量
df_init = pd.read_csv(Path(input_dir, "input_label.csv"))
df_anno = None if is_first_train else pd.read_csv(Path(input_dir, "input_added_label.csv"))
df_img = pd.read_csv(Path(input_dir, "input_unlabel.csv"))
model_dir_upload = str(Path(output_dir, "model_upload").resolve())
model_dir_check = str(Path(output_dir, "model_check").resolve())
work_dir = str(Path(output_dir, "work_dir").resolve())
result_csv_path = str(Path(output_dir, "output_query.csv").resolve())

import shutil

shutil.rmtree(output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ===============================================
# 保存待上传的模型
model = MyModel()
main_utils.save_pipeline_model(model=model, model_id=model_id, save_dir=model_dir_upload)

# ===============================================
# 检查模型功能是否正常

# split train val
df_train, df_val = model.split_train_val(df_init, df_anno, is_first_train, val_size=val_size, random_seed=1)

# train
model.train_model(df_train=df_train, df_val=df_val, work_dir=work_dir, is_first_train=is_first_train, **options)

# query
df_result = model.query_hard_example(df_img, work_dir=work_dir, query_cnt=query_cnt, strategy=strategy)

# save result csv
Path(result_csv_path).parent.mkdir(parents=True, exist_ok=True)
df_result.to_csv(result_csv_path, index=False)

# persist pipeline
main_utils.save_pipeline_model(model=model, model_id=model_id, save_dir=model_dir_check)

# ===============================================
print("\n\n\n如果检查结果正常,可以把生成的模型打包上传")
print("需要打包上传的Pipeline模型文件是: {}".format(model_dir_upload))
print("\n\nDone!\n\n")
