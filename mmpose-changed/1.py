#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import torch
import random
import shutil
import deal_ds
import pandas as pd
from time import time
from pathlib import Path
from datacanvas.aps import dc
from torch import distributed as dist
from tools.train import get_training_cfg
from tools.test import kpt_eval
from model_nets.utils import evaluate
from model_nets.dekr_hrnet_estimator import DekrHrnetEstimator

# 取local_rank,取不到就填写0
local_rank = int(os.environ.get("LOCAL_RANK", 0))
# 在哪一个进程
RANK = int(os.environ.get("RANK"))
# 设定cuda指定的GPU卡号，这样model可以放到指定local_rank的GPU上，而不会重复
device = 'cuda' if torch.cuda.is_available() else 'cpu'
world_size = int(os.environ.get("WORLD_SIZE", 1))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

model_weight = "/opt/modelrepo/public/964f160d-14d3-11ee-9446-e20af64b9f79/pretrained_dir/hrnet_w32_coco_512x512-867b9659_20220928.pth"

work_dir = Path(dc.conf.global_params.work_dir, dc.conf.global_params.block_id)
Path(work_dir).mkdir(parents=True, exist_ok=True)
work_dir = str(work_dir)
print("work_dir:", work_dir)

# 创建训练后权重的保存位置
best_model_dir = str(dc.conf.outputs.best_model_dir)
latest_model_dir = str(dc.conf.outputs.latest_model_dir)
Path(best_model_dir).mkdir(parents=True, exist_ok=True)
Path(latest_model_dir).mkdir(parents=True, exist_ok=True)
tensorboard_dir = str(dc.conf.outputs.tensorboard_dir)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
# mkdir
train_data_path = os.path.join(work_dir, 'train2017')
val_data_path = os.path.join(work_dir, 'val2017')
annos_path = os.path.join(work_dir, 'annotations')
# clear
if RANK == 0:
    if os.path.exists(train_data_path):
        shutil.rmtree(train_data_path)
    if os.path.exists(val_data_path):
        shutil.rmtree(val_data_path)
    if os.path.exists(annos_path):
        shutil.rmtree(annos_path)

Path(val_data_path).mkdir(parents=True, exist_ok=True)
Path(train_data_path).mkdir(parents=True, exist_ok=True)
Path(annos_path).mkdir(parents=True, exist_ok=True)

# 创建进程组并指定后端通信的模块，仅CPU的话只能使用gloo通信，至少两个个进程，每个进程至少一个GPU
if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
    dist.init_process_group(backend="nccl", world_size=int(os.environ.get("WORLD_SIZE", 1)),
                            rank=int(os.environ.get("RANK")))
else:
    # 仅cpu或者一个进程一个GPU的情况
    dist.init_process_group(backend="gloo", world_size=int(os.environ.get("WORLD_SIZE", 1)),
                            rank=int(os.environ.get("RANK")))

# 读取参数
lr = float(dc.conf.params.lr)
cosine_lr = str(dc.conf.params.cosine_lr) == "True"
batch_size = int(dc.conf.params.batch_size)
total_epoch = int(dc.conf.params.total_epoch)
optimizer = str(dc.conf.params.optimizer)
kpt_thr = float(dc.conf.params.kpt_thr)
freeze_layers = str(dc.conf.params.freeze_layers) == "True"
val_size = float(dc.conf.params.val_size)
use_amp = str(dc.conf.params.use_amp) == "True"
if device == 'cpu':
    use_amp = False
# 使用梯度压缩，不能和混合精度一起使用
if world_size > 1 and device == 'cuda':
    use_amp = False

# 读取数据--得到合并后的coco对象ds.data
ds = dc.dataset(dc.conf.inputs.image_data).read()
df_all, unlabeled_imgids = deal_ds.get_df(ds, seed=1)
assert len(df_all.iloc[0, 1][0]) % 3 == 0, 'One keypoint must be in the format [x, y, visibility]!'
num_keypoints = len(df_all.iloc[0, 1][0]) // 3

coco = ds.data
imgIds = coco.getImgIds()
imgIds = list(set(imgIds) - set(unlabeled_imgids))
df_all.index = imgIds

random.seed(1)
random.shuffle(imgIds)
# 计算训练集和验证集的数量
num_val = int(len(imgIds) * val_size)
num_train = len(imgIds) - num_val
list_train = imgIds[:num_train]
list_val = imgIds[num_train:]

# 获取所有标签，图片和类别信息
anns = coco.loadAnns(coco.getAnnIds())
imgs = coco.loadImgs(imgIds)
cats = coco.loadCats(coco.getCatIds())

# 保存数据信息到对应的目录以及创建临时目录等
if RANK == 0:
    import multiprocessing

    t = time()
    params = []
    # 创建进程池
    env_var = int(os.environ.get('OMP_NUM_THREADS'))  # number of cpu/workers
    pool = multiprocessing.Pool(env_var)

    for i, img in enumerate(imgs):
        if i < num_train:
            params.append((img, train_data_path))
        else:
            params.append((img, val_data_path))

    # 将计算任务分配到多个进程中，并获取计算结果
    results = pool.map(deal_ds.images_split_copy, params)

    # 关闭进程池
    pool.close()
    pool.join()

    print(f'数据整理总共耗时：{time() - t:.3f}s.\n')

    # 拆分数据集
    train_data_dict = {
        'images': imgs[:num_train],
        'annotations': [ann for ann in anns if ann['image_id'] in list_train],
        'categories': cats
    }

    val_data_dict = {
        'images': imgs[num_train:],
        'annotations': [ann for ann in anns if ann['image_id'] in list_val],
        'categories': cats
    }

    # 生成测试的df
    test_data = df_all.loc[list_val]

    # 保存拆分后的数据集的标注信息
    with open(annos_path + '/person_keypoints_train2017.json', 'w') as f:
        json.dump(train_data_dict, f)

    with open(annos_path + '/person_keypoints_val2017.json', 'w') as f:
        json.dump(val_data_dict, f)

dist.barrier()

# 获取train_cfg并更新配置 train_cfg is a tuple included
# (model, datasets, cfg, distributed, not args.no_validate, timestamp, meta)
train_cfg = get_training_cfg(work_dir, model_weight, batch_size, total_epoch, use_amp, num_keypoints)
model = train_cfg[0]
cfg = train_cfg[2]

if optimizer == 'SGD':
    cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
if cosine_lr:
    cfg.lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1.0 / 10,
        min_lr_ratio=1e-5)
if freeze_layers:
    for param in model.backbone.parameters():
        param.requires_grad = False
cfg.optimizer.lr = lr

# 2.训练模型
dekr_hrnet_estimator = DekrHrnetEstimator(input_cols=['img'], target_cols=['target'], output_cols=['prediction'])
dekr_hrnet_estimator.lr = lr
dekr_hrnet_estimator.total_epoch = total_epoch
dekr_hrnet_estimator.optimizer = optimizer
dekr_hrnet_estimator.Cosine_lr = cosine_lr
dekr_hrnet_estimator.batch_size = batch_size
dekr_hrnet_estimator.work_dir = work_dir
dekr_hrnet_estimator.tensorboard_dir = tensorboard_dir
dekr_hrnet_estimator.best_model_dir = best_model_dir
dekr_hrnet_estimator.latest_model_dir = latest_model_dir
dekr_hrnet_estimator.use_amp = use_amp
dekr_hrnet_estimator.freeze_layers = freeze_layers
dekr_hrnet_estimator.device = device
dekr_hrnet_estimator.cfg = cfg
dekr_hrnet_estimator.kpt_thr = kpt_thr
dekr_hrnet_estimator.fit(*train_cfg)

# predict
if int(os.environ.get("RANK")) == 0:
    # 整理训练结果
    latest_path = work_dir + '/latest.pth'
    tf_path = work_dir + '/tf_logs'

    if os.path.exists(latest_path):
        shutil.copy2(latest_path, latest_model_dir + '/model-latest.pth')
    for filename in os.listdir(work_dir):
        if filename.startswith('best'):
            shutil.copy2(os.path.join(work_dir, filename), best_model_dir + '/model-best.pth')
    if not os.path.exists(best_model_dir + '/model-best.pth') and os.path.exists(latest_path):
        shutil.copy2(latest_path, best_model_dir + '/model-best.pth')
    if os.path.exists(tf_path):
        for tf in os.listdir(tf_path):
            shutil.move(os.path.join(tf_path, tf), os.path.join(tensorboard_dir, tf))
        os.rmdir(tf_path)
        print('rm tf_logs done!')

    # predicting
    test_data = df_all.loc[list_val]
    x_test = test_data[['path']]
    predictions = dekr_hrnet_estimator.predict(x_test, is_need_create_model=True)
    test_data = test_data.reset_index(drop=True)
    predictions = pd.concat([test_data, predictions], axis=1)

    # prediction update
    dc.dataset(dc.conf.outputs.prediction).update(predictions)

    # 评估
    weights_dir = best_model_dir + '/model-best.pth'
    print('-------------Starting Evaluating------------------')

    eval_results = kpt_eval(cfg, weights_dir, work_dir)
    performance = evaluate(eval_results)
    # 输出结果 performance
    with open(dc.conf.outputs.performance, 'w') as f:
        json.dump(performance, f)

    print('--------------Evaluating Done------------------')
    exit()
    dekr_hrnet_estimator.persist()
    # 删除用于保存训练过程中产生中间结果的目录
    shutil.rmtree(work_dir)

dc.logger.info("done.")
