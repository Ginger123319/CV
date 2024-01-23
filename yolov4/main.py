#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.aps import dc

from model_nets.yolov4_estimator import YoloV4Estimator
from script_files import training_utils, evaluating

import os
import json
import shutil
import pandas as pd

from pathlib import Path
work_dir = Path(dc.conf.global_params.work_dir, dc.conf.global_params.block_id)
Path(work_dir).mkdir(parents=True, exist_ok=True)
work_dir = str(work_dir)
print("work_dir:", work_dir)

os.system('mkdir ' + dc.conf.outputs.model_dir)
pth_logs_dir = str(dc.conf.outputs.model_dir)
tensorboard_dir = str(dc.conf.outputs.train_logs)

# 读取参数
classes = dc.conf.params.classes.split(',')
image_size = int(dc.conf.params.input_shape)
create_new_anchors = str(dc.conf.params.create_new_anchors) == "True"
lr = float(dc.conf.params.lr)
cosine_lr = str(dc.conf.params.cosine_lr) == "True"
mosaic = str(dc.conf.params.mosaic) == "True"
smooth_label = str(dc.conf.params.smooth_label) == "True"
batch_size = int(dc.conf.params.batch_size)
freeze_epoch = int(dc.conf.params.freeze_epoch)
total_epoch = int(dc.conf.params.total_epoch)
optimizer = str(dc.conf.params.optimizer)
n_weights_saved = int(dc.conf.params.num_of_weights)
drop_block = float(dc.conf.params.drop_block)
val_size = float(dc.conf.params.val_size)
confidence = float(dc.conf.params.confidence)
iou = float(dc.conf.params.iou)
use_tfrecord = str(dc.conf.params.use_tfrecord)=="True"
use_amp = str(dc.conf.params.use_amp)=="True"
test_size = float(dc.conf.params.test_size)

# 读取数据
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

# 保存预训练权重
weight_path = "/opt/modelrepo/public/9829c603-1960-11ed-b2a3-3c2c30c2a5f3/pretrained_dir/yolo4_coco_weights.pth"
training_utils.load_pre_training_weights(weight_path, work_dir)

# 生成用于保存训练过程中产生中间结果的目录
training_utils.make_middle_dir(work_dir)

# 生成数据集标注信息文件、类别信息文件
training_utils.create_train_info_txt(train_data, classes, work_dir)
training_utils.create_classes_txt(classes, work_dir)

# 2.训练模型
yolov4_estimator = YoloV4Estimator(input_cols=['path'], target_cols=['target'], output_cols=['prediction'])
yolov4_estimator.num_classes = len(classes)
yolov4_estimator.image_size = image_size
yolov4_estimator.val_size = val_size
yolov4_estimator.lr = lr
yolov4_estimator.freeze_epoch = freeze_epoch
yolov4_estimator.total_epoch = total_epoch
yolov4_estimator.optimizer = optimizer
yolov4_estimator.Cosine_lr = cosine_lr
yolov4_estimator.mosaic = mosaic
yolov4_estimator.smooth_label = smooth_label
yolov4_estimator.drop_block = drop_block
yolov4_estimator.n_weights_saved = n_weights_saved
yolov4_estimator.batch_size = batch_size
yolov4_estimator.work_dir = work_dir
yolov4_estimator.pth_logs_dir = pth_logs_dir
yolov4_estimator.tensorboard_dir = tensorboard_dir
yolov4_estimator.use_tfrecord = use_tfrecord
yolov4_estimator.use_amp = use_amp

yolov4_estimator.fit(x_train, y_train)
predictions = yolov4_estimator.predict2(x_test)
predictions = pd.concat([test_data, predictions], axis=1)
yolov4_estimator.persist()

# 评估
best_model = evaluating.get_gt_dr_map(test_data, image_size, classes, work_dir)
performance = evaluating.evaluate(best_model, work_dir)

# 输出结果
# performance
with open(dc.conf.outputs.performance, 'w') as f:
    json.dump(performance, f)

# prediction
dc.dataset(dc.conf.outputs.prediction).update(predictions)

# best_model_dir
os.system('mkdir ' + dc.conf.outputs.best_model_dir)
shutil.copyfile(work_dir+'/middle_dir/normal_train_best_model_dir/'+best_model, dc.conf.outputs.best_model_dir+'/'+best_model)

# 删除用于保存训练过程中产生中间结果的目录
training_utils.destroy_middle_dir(work_dir)
shutil.rmtree(work_dir+"/pre_training_weights")

dc.logger.info("done.")

