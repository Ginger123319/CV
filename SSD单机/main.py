#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.aps import dc
from model_nets.ssd_estimator import SSDEstimator
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
os.system('mkdir ' + dc.conf.outputs.best_model_dir)
pth_logs_dir = str(dc.conf.outputs.model_dir)
tensorboard_dir = str(dc.conf.outputs.train_logs)

# 读取参数
classes = dc.conf.params.classes.split(',')
image_size = 300
lr = float(dc.conf.params.lr)
batch_size = int(dc.conf.params.batch_size)
freeze_epoch = int(dc.conf.params.freeze_epoch)
total_epoch = int(dc.conf.params.total_epoch)
optimizer = str(dc.conf.params.optimizer)
n_weights_saved = int(dc.conf.params.num_of_weights)
val_size = float(dc.conf.params.val_size)
confidence = float(dc.conf.params.confidence)
iou = float(dc.conf.params.iou)
use_tfrecord = str(dc.conf.params.use_tfrecord) == "True"
use_amp = str(dc.conf.params.use_amp) == "True"
mosaic = str(dc.conf.params.mosaic) == "True"
test_size = float(dc.conf.params.test_size)

# 自动数据增强参数
image_augment = dc.conf.params.image_augment
population_size = int(dc.conf.params.population_size)
auto_aug_ratio = float(dc.conf.params.auto_aug_ratio)
auto_aug_epochs = total_epoch if str(dc.conf.params.auto_aug_epochs).strip() in ["", "None"] else int(dc.conf.params.auto_aug_epochs)


# 读取数据
#train_data = dc.dataset(dc.conf.inputs.train_data).read()
#test_data = dc.dataset(dc.conf.inputs.test_data).read()
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
weight_path = "/opt/modelrepo/public/9829c603-1960-11ed-b2a3-3c2c30c2a5f3/pretrained_dir/ssd_weights.pth"
training_utils.load_pre_training_weights(weight_path, work_dir)

# 生成用于保存训练过程中产生中间结果的目录
training_utils.make_middle_dir(work_dir)

# 生成数据集标注信息文件、类别信息文件
training_utils.create_train_info_txt(train_data, classes, work_dir, filename="train_info.txt")
if image_augment=="PBA":
    if auto_aug_ratio<1:
        pba_data = train_data.sample(frac=auto_aug_ratio, axis=0, replace=False, random_state=1).reset_index(drop=True)
    else:
        pba_data = train_data
    training_utils.create_train_info_txt(pba_data, classes, work_dir, filename="pba_info.txt")

training_utils.create_classes_txt(classes, work_dir)

# 2.训练模型
ssd_estimator = SSDEstimator(input_cols=['path'], target_cols=['target'], output_cols=['prediction'])
ssd_estimator.num_classes = len(classes)
ssd_estimator.image_size = image_size
ssd_estimator.val_size = val_size
ssd_estimator.lr = lr
ssd_estimator.freeze_epoch = freeze_epoch
ssd_estimator.total_epoch = total_epoch
ssd_estimator.optimizer = optimizer
ssd_estimator.n_weights_saved = n_weights_saved
ssd_estimator.batch_size = batch_size
ssd_estimator.work_dir = work_dir
ssd_estimator.pth_logs_dir = pth_logs_dir
ssd_estimator.tensorboard_dir = tensorboard_dir
ssd_estimator.confidence = confidence
ssd_estimator.iou = iou
ssd_estimator.use_tfrecord = use_tfrecord
ssd_estimator.use_amp = use_amp
# Auto augment config
ssd_estimator.image_augment = image_augment
ssd_estimator.population_size = population_size
ssd_estimator.auto_aug_ratio = auto_aug_ratio 
ssd_estimator.auto_aug_epochs = auto_aug_epochs
ssd_estimator.annotation_path = work_dir+'/middle_dir/data_info/train_info.txt'
ssd_estimator.mosaic = mosaic


population_size = int(dc.conf.params.population_size)
ssd_estimator.population_size = population_size

ssd_estimator.fit(x_train, y_train)


def remove_file(dir_path, file_pattern):

    from pathlib import Path

    g = Path(dir_path).rglob(file_pattern)
    for f in g:
        if f.is_file():
            print("Removing tmp file... {}".format(f.absolute()))
            f.unlink()
remove_file(tensorboard_dir, "*.h5")

predictions = ssd_estimator.predict2(x_test)
predictions = pd.concat([test_data, predictions], axis=1)
ssd_estimator.persist()

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
shutil.copyfile(work_dir+'/middle_dir/normal_train_best_model_dir/'+best_model, dc.conf.outputs.best_model_dir+'/'+best_model)

# 删除用于保存训练过程中产生中间结果的目录
training_utils.destroy_middle_dir(work_dir)
shutil.rmtree(work_dir+"/pre_training_weights")


dc.logger.info("done.")

