#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.aps import dc 
from model_nets.mask_rcnn_estimator import MaskRCNNEstimator 
from script_files import training_utils, evaluating

import os 
import json
import shutil
import pandas as pd 
import torch
from torch import distributed as dist 


work_dir = dc.conf.global_params.work_dir
print("work_dir:", work_dir)

if int(os.environ.get("RANK")) == 0:
    os.system('mkdir ' + dc.conf.outputs.model_dir)
pth_logs_dir = str(dc.conf.outputs.model_dir)
tensorboard_dir = str(dc.conf.outputs.train_logs)

if torch.cuda.is_available():
    dist.init_process_group(backend="nccl", world_size=int(os.environ.get("WORLD_SIZE", 1)), rank=int(os.environ.get("RANK")))
else:
    dist.init_process_group(backend="gloo", world_size=int(os.environ.get("WORLD_SIZE", 1)), rank=int(os.environ.get("RANK")))

# 读取参数
classes = dc.conf.params.classes.split(',')
image_size = int(dc.conf.params.input_shape)
val_size = float(dc.conf.params.val_size)
lr = float(dc.conf.params.lr)
freeze_epoch = int(dc.conf.params.freeze_epoch)
total_epoch = int(dc.conf.params.total_epoch)
optimizer = dc.conf.params.optimizer
batch_size = int(dc.conf.params.batch_size)
n_weights_saved = int(dc.conf.params.num_of_weights)
use_tfrecord = str(dc.conf.params.use_tfrecord) == "True"
use_amp = str(dc.conf.params.use_amp) == "True"
mosaic = str(dc.conf.params.mosaic) == "True"
test_size = float(dc.conf.params.test_size)

# 读入数据
# train_data = dc.dataset(dc.conf.inputs.train_data).read()
# test_data = dc.dataset(dc.conf.inputs.test_data).read()

import deal_ds
ds = dc.dataset(dc.conf.inputs.image_data).read()
train_data, test_data = deal_ds.get_df(ds, test_size, seed=1, classes=classes)


train_data = train_data.reset_index(drop=True)
x_train = train_data[['path']]
y_train = train_data['target']

test_data = test_data.reset_index(drop=True)
x_test = test_data[['path']]
y_test = test_data['target']


if int(os.environ.get("RANK")) == 0:
    # 保存预训练权重
    weight_path = "/opt/modelrepo/public/9829c603-1960-11ed-b2a3-3c2c30c2a5f3/pretrained_dir/voc_weights_resnet.pth"
    training_utils.load_pre_training_weights(weight_path, work_dir)
    
    # 生成用于保存训练过程中产生中间结果的目录
    training_utils.make_middle_dir(work_dir)
    
    # 生成数据集标注信息文件、类别信息文件
    training_utils.create_train_info_txt(train_data, classes, work_dir)
    training_utils.create_classes_txt(classes, work_dir)
dist.barrier()    

# 训练模型
maskrcnn_estimator = MaskRCNNEstimator(input_cols=['path'], target_cols=['target'], output_cols=['prediction'])
maskrcnn_estimator.num_classes = len(classes)
maskrcnn_estimator.image_size = image_size
maskrcnn_estimator.val_size = val_size
maskrcnn_estimator.lr = lr
maskrcnn_estimator.freeze_epoch = freeze_epoch
maskrcnn_estimator.total_epoch = total_epoch
maskrcnn_estimator.optimizer = optimizer
maskrcnn_estimator.n_weights_saved = n_weights_saved
maskrcnn_estimator.batch_size = batch_size
maskrcnn_estimator.work_dir = work_dir
maskrcnn_estimator.pth_logs_dir = pth_logs_dir
maskrcnn_estimator.tensorboard_dir = tensorboard_dir
maskrcnn_estimator.dist = dist
maskrcnn_estimator.use_tfrecord = use_tfrecord
maskrcnn_estimator.use_amp = use_amp
maskrcnn_estimator.mosaic = mosaic

maskrcnn_estimator.fit(x_train, y_train)

if int(os.environ.get("RANK")) == 0:
    predictions = maskrcnn_estimator.predict2(x_test)
    predictions = pd.concat([test_data, predictions], axis=1)
    maskrcnn_estimator.persist()

    # 评估
    best_model = evaluating.get_gt_dr_map(test_data, image_size, classes, work_dir)
    performance = evaluating.evaluate(best_model, work_dir)

    #输出结果
    with open(dc.conf.outputs.performance, 'w') as f:
        json.dump(performance, f)
    
    # prediction
    dc.dataset(dc.conf.outputs.prediction).update(predictions)   

    # best_model_dir
    os.system('mkdir ' + dc.conf.outputs.best_model_dir)
    shutil.copyfile(work_dir+'/middle_dir_mask_dis/normal_train_best_model_dir/'+best_model, dc.conf.outputs.best_model_dir+'/'+best_model)
    # 删除用于保存训练过程中产生中间结果的目录
    training_utils.destroy_middle_dir(work_dir)
    shutil.rmtree(work_dir+"/pre_training_weights_mask")
    

dc.logger.info("done.")

