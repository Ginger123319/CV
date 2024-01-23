#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import ast
import pandas as pd
from pathlib import Path
from datacanvas.aps import dc
from main_utils import check_data
from main_utils import show_gpu_info
from dc_model_repo import model_repo_client
from main_utils import load_pipeline_model, Model, save_pipeline_model

dc.logger.info("dc is ready!")
# show gpu info
show_gpu_info()

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
is_need_create_model = params_in["is_need_create_model"]
query_strategy = str(params_in["strategy"])
query_num = int(params_in['query_cnt'])
partition_dir = params_in['partition_dir']

is_first_train = True if is_first_train.lower() == "true" else False
is_need_create_model = True if is_need_create_model.lower() == "true" else False

# 训练预测使用的参数
CONFIG = {
    "optimizer": options['optimizer'],
    'fit_batch_size': int(options['fit_batch_size']),
    "FIT_epochs": int(options["fit_epoch"]),
    "input_shape": int(options["input_shape"]),
    "image_col": image_col,
    "label_col": label_col,
    "val_ratio": val_ratio,
    "id_col": options["id_col"],
    "partition_dir": partition_dir
}
# 读取未标注数据
df_unlabel = pd.read_csv(input_unlabel_path)

# download, load and prepare pipeline
model = load_pipeline_model(model_url=model_url, work_dir=work_dir)
assert isinstance(model, Model)

if is_need_create_model:
    # input_data
    df_label = pd.read_csv(input_label_path)
    df_hard = None if is_first_train else pd.read_csv(input_added_label_path)

    # 对标签列进行处理
    df_label.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                  df_label.loc[:, label_col]]
    # 对标签列进行处理
    if df_hard is not None:
        df_hard.loc[:, label_col] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in
                                     df_hard.loc[:, label_col]]
        df_cls = pd.concat([df_label, df_hard], axis=0, ignore_index=True)
    else:
        df_cls = df_label
    # 检查数据
    check_data(df_label, label_col)
    # 目前类别数目、类别名称
    class_num = df_cls[label_col].nunique()
    class_name = list(set(df_cls[label_col].tolist()))
    class_name.sort()

    CONFIG["class_name"] = class_name
    CONFIG["class_num"] = class_num
    CONFIG["df_unlabeled"] = df_unlabel
    dc.logger.info("CONFIG:\n {}".format(repr(CONFIG)))

    # split train val
    df_train, df_val = model.split_train_val(df_label, df_hard, is_first_train, val_size=val_ratio)

    # train
    model.train_model(df_train=df_train, df_val=df_val, work_dir=work_dir, is_first_train=is_first_train, **CONFIG)

    # query
    df_result = model.query_hard_example(df_unlabel, work_dir=work_dir, query_cnt=query_num, strategy=query_strategy, **CONFIG)

    # save result csv
    df_result.to_csv(result_path, index=False)

    # persist pipeline
    save_pipeline_model(model=model, model_id=target_model_Id, save_dir=target_model_path)
else:
    class_name = model.class_name
    CONFIG["class_name"] = class_name
    dc.logger.info("CONFIG:\n {}".format(repr(CONFIG)))

    # predict
    df_result = model.predict(df_unlabel, work_dir=work_dir, **CONFIG)

    # save result csv
    df_result.to_csv(result_path, index=False)

dc.logger.info("done.")
