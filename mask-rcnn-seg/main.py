import os
import pandas as pd
import json
import shutil
import torch 
import re 

from tfrecord.tools import create_index
from script_files import tfrecord_split, create_tfrecord
from model_nets.mask_estimator import MaskRCNNEstimator
from datacanvas.aps import dc 
dc.logger.info("dc is ready!")


val_ratio = float(dc.conf.params.val_ratio)

# 读入数据
dataset_train = dc.dataset(dc.conf.inputs.train_dataset).read()

ds = dc.dataset(dc.conf.inputs.instance_segmentation_dataset)
simple = ds.read()
train_dataset, val_dataset = simple.split(ratio=1-val_ratio, type='order')
dataset_type = simple.label_format # VOC 或者 COCO

# 创建算子的输出目录
work_dir = dc.conf.global_params.work_dir + "/maskrcnn_seg_alone"
print("work_dir:", work_dir)

shutil.rmtree(work_dir, ignore_errors=True)
os.makedirs(work_dir, exist_ok=True) 
os.makedirs(str(dc.conf.outputs.model_dir), exist_ok=True) # 创建pth文件输出目录
os.makedirs(str(dc.conf.outputs.train_logs_dir), exist_ok=True) # 创建训练日志输出目录
os.makedirs(str(dc.conf.outputs.prediction), exist_ok=True) # 创建预测结果输出目录
os.makedirs(os.path.join(str(dc.conf.outputs.prediction), 'original_image'), exist_ok=True)
os.makedirs(os.path.join(str(dc.conf.outputs.prediction), 'mask_image'), exist_ok=True)
os.makedirs(os.path.join(str(dc.conf.outputs.prediction), 'prediction_image'), exist_ok=True)

# 读入参数
pretrained_backbone = str(dc.conf.params.pretrained_backbone) == 'True' #
trainable_backbone_layers = int(dc.conf.params.trainable_backbone_layers)
confidence = float(dc.conf.params.confidence)
step_lr = float(dc.conf.params.step_lr)
step_weight_decay = float(dc.conf.params.step_weight_decay)
total_epoch = int(dc.conf.params.total_epoch)
max_trials = int(dc.conf.params.max_trials)
early_stop = int(dc.conf.params.early_stop)
tuning_strategy = str(dc.conf.params.tuning_strategy)
lr = str(dc.conf.params.lr).split(',')
batch_size = str(dc.conf.params.batch_size).split(',')
optimizer = str(dc.conf.params.optimizer).split(',')
weight_decay = str(dc.conf.params.weight_decay).split(',')
activation_function = str(dc.conf.params.activation_function).split(',')
class_name = str(dc.conf.params.class_name).split(',')
class_name.insert(0, 'background')
print("class_name:", class_name)

pretrained_pth = "/opt/modelrepo/public/128c0114-38b4-11ed-b51a-1c1bb5ce0b40/pretrained_dir/maskrcnn_coco.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = str(dc.conf.outputs.model_dir)
tensorboard_dir = str(dc.conf.outputs.train_logs_dir)
performance_path = str(dc.conf.outputs.performance)

# 实例化Estimator
mask_estimator = MaskRCNNEstimator(input_cols=['image_path'], target_cols=None, output_cols=["prediction"])

mask_estimator.class_name = class_name
mask_estimator.trainable_backbone_layers = trainable_backbone_layers
mask_estimator.pretrained_backbone = pretrained_backbone
mask_estimator.confidence = confidence
mask_estimator.step_lr = step_lr
mask_estimator.step_weight_decay = step_weight_decay

mask_estimator.total_epoch = total_epoch
mask_estimator.max_trials = max_trials
mask_estimator.early_stop = early_stop
mask_estimator.tuning_strategy = tuning_strategy
mask_estimator.lr = lr
mask_estimator.batch_size = batch_size
mask_estimator.optimizer = optimizer
mask_estimator.weight_decay = weight_decay
mask_estimator.activation_function = activation_function

mask_estimator.pretrained_pth = pretrained_pth
mask_estimator.device = device
mask_estimator.model_dir = model_dir
mask_estimator.tensorboard_dir = tensorboard_dir
mask_estimator.performance_path = performance_path
mask_estimator.work_dir = work_dir
mask_estimator.dataset_type = dataset_type

mask_estimator.fit(X=dataset_train, train_dataset=train_dataset, val_dataset=val_dataset)

mask_estimator.persist()

# 加载模型并在测试集上进行预测
if dataset_type == "COCO":
    data_list = []
    image_ids = list(val_dataset.data.imgToAnns.keys())
    for image_id in image_ids:
        image_path = val_dataset.data.loadImgs(image_id)[0]["file_full_path"]
        data_list.append(image_path)
    test_dataset = pd.DataFrame(data_list, columns=['image_path'])
    
elif dataset_type == "VOC":
    data_list = []
    for i in range(val_dataset.length):
        image_path = val_dataset.get_item(i).data
        data_list.append(image_path)
    test_dataset = pd.DataFrame(data_list, columns=['image_path'])    
else:
    raise Exception("do not support this dataset type!")
mask_estimator.predict_local(test_dataset)

shutil.rmtree(work_dir, ignore_errors=True)
print("work_dir has been destroyed!")

dc.logger.info("Done!")


