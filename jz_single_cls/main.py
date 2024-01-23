#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datacanvas.aps import dc
from dc_model_repo import model_repo_client
import pandas as pd
import os
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
import utils
import torch
from collections import Counter
import sys
import shutil

# put your code here
dc.logger.info("dc is ready!")

# work_dir
work_dir = model_repo_client.get_dc_proxy().get_work_dir()
work_dir = os.path.join(work_dir, dc.conf.global_params.block_id, "work_files")
Path(work_dir).mkdir(parents=True, exist_ok=True)
print("Current work_dir: {}".format(work_dir))

# input params
dc.logger.info("Input params:")
params_in = {}
for k, v in dc.conf.params._asdict().items():
    dc.logger.info("{}: {}".format(k, repr(v)))
    params_in[k] = str(v).strip()

# 读入参数中的目标值
options = params_in["options"]
options = ast.literal_eval(options)

image_col = options["image_col"]
label_col = options["label_col"]
val_ratio = float(params_in["val_size"])
model_url = params_in["model_url"]
input_label_path = params_in["input_label_path"]
input_added_label_path = params_in["input_added_label_path"]
input_unlabel_path = params_in["input_unlabel_path"]

result_path = params_in["result_path"]
target_model_Id = params_in["target_model_Id"]
target_model_path = params_in["target_model_path"]
is_first_train = params_in["is_first_train"]
query_strategy = str(params_in["strategy"])
query_num = int(params_in['query_cnt'])
partition_dir = params_in['partition_dir']

is_first_train = True if is_first_train.lower() == "true" else False

# input_data
df_label = pd.read_csv(input_label_path)
df_unlabel = pd.read_csv(input_unlabel_path)

# 对标签列进行处理
df_label.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                              df_label.loc[:, label_col]]

c = Counter(df_label[label_col])
if len(c) < 2:
    raise Exception("分类的类别至少是2，现在为：{}".format(len(c)))
break_flag = False
for k, v in c.items():
    # print("类别[{}]个数：{}".format(k, v))
    if v < 2:
        break_flag = True
if break_flag:
    raise Exception("每个类别的样本数至少为2！")

if val_ratio != 0.0:
    # 拆分数据集
    df_train, df_val = train_test_split(df_label, test_size=val_ratio, random_state=1, shuffle=True,
                                        stratify=df_label["label"])
else:
    df_train = df_label
    df_val = None

# 训练集合并难例数据
if not is_first_train:
    df_hard = pd.read_csv(input_added_label_path)
    df_hard.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                 df_hard.loc[:, label_col]]
    # 合并难例和初始的训练样本
    df_train = pd.concat([df_train, df_hard], axis=0, ignore_index=True)
    # 合并所有样本，便于统计新的类别数目
    df_label = pd.concat([df_label, df_hard], axis=0, ignore_index=True)

# 目前类别数目
class_num = df_label[label_col].nunique()

# 统计训练集中类别名称
class_name = list(set(df_label[label_col].tolist()))
class_name.sort()

load_pipeline = True
if load_pipeline:
    from utils import load_pipeline_model

    model = load_pipeline_model(model_url=model_url, work_dir=work_dir)
    from classifier_multi import Model

    assert isinstance(model, Model)
    if is_first_train:
        model.adjust_model(class_num=class_num)
    else:
        if model.out_dim != class_num:
            sys.exit("expected class_num: {} but get class_num: {}".format(model.out_dim, class_num))
else:
    from classifier_multi.my_model import MyModel

    model = MyModel(name='resnet50')

# 训练预测使用的参数
CONFIG = {
    "optimizer": options['optimizer'],
    'fit_batch_size': int(options['fit_batch_size']),
    "FIT_epochs": int(options["fit_epoch"]),
    "class_name": class_name,
    "input_shape": int(options["input_shape"]),
    "image_col": image_col,
    "label_col": label_col,
    "val_ratio": val_ratio,
    "id_col": options["id_col"],
    "partition_dir": partition_dir
}

dc.logger.info("CONFIG:\n {}".format(repr(CONFIG)))

# # 开始训练
# model.load_weights()

# train
model.train_model(df_train=df_train, df_val=df_val, net_config=CONFIG)

# predict
pred = model.predict(df_unlabel, CONFIG)

# 预测结果后处理
df_unlabel.drop(image_col, axis=1, inplace=True)
# 确定预测的类别
out = torch.argmax(pred, dim=1).tolist()
label = [{"annotations": [{"category_id": class_name[index]}]} for index in out]
df_unlabel[label_col] = label
df_unlabel['isHardSample'] = pred
df_pred = df_unlabel

# select
df_result = utils.select_hard_example(df_pred, query_strategy, query_num)

# save result csv
df_result.to_csv(result_path, index=False)

# persist pipeline
utils.save_pipeline_model(model=model, model_id=target_model_Id, save_dir=target_model_path)

shutil.rmtree(Path(work_dir, "tmp"), ignore_errors=True)

dc.logger.info("done.")
