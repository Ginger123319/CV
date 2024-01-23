#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datacanvas.aps import dc
from dc_model_repo import model_repo
from dc_model_repo import model_repo_client
import pickle
import json
import pandas as pd
import numpy as np
from pba.image_classifier import ImageClassificationEstimator
import h5py
import os
from pathlib import Path
dc.logger.info("dc is ready!")

dc.logger.info("Input params:")
params_in = {}
for k, v in dc.conf.params._asdict().items():
    dc.logger.info("{}: {}".format(k, repr(v)))
    params_in[k] = str(v).strip()

work_dir = model_repo_client.get_dc_proxy().get_work_dir()
work_dir = os.path.join(work_dir, dc.conf.global_params.block_id, "work_files")
Path(work_dir).mkdir(parents=True, exist_ok=True)
print("Current work_dir: {}".format(work_dir))

# 读入参数中的目标值

image_col = params_in["image_col"]
label_col = params_in["label_col"]
output_cols = [params_in["predict_col"]]

test_size = float(params_in["test_size"])
ds = dc.dataset(dc.conf.inputs.image_data).read()
import deal_ds
df_train, df_test = deal_ds.get_df(ds, test_size, seed=int(params_in["random_seed"]), image_col=image_col, label_col=label_col)

code_dir = dc.conf.global_params.code_dir
# pretained_weights = os.path.join(code_dir, "pretrained_weights")
pretained_weights = "/opt/modelrepo/public/133b23fe-0281-11ed-b678-3c2c30c2a5f3/pretrained_dir/pretrained_weights_tf2"
tensorboard_dir = str(dc.conf.outputs.tensorboard_dir)


class_num = df_train[label_col].nunique()

auto_aug_ratio = float(params_in["auto_aug_ratio"])
assert 0<auto_aug_ratio<=1, "auto_aug_ratio需要是大于0小于等于1的数字。"

auto_aug_config = None if params_in["auto_aug_config"] in ["None", ""] else params_in["auto_aug_config"]
FIT_epochs = int(params_in["FIT_epochs"])
auto_aug_epochs = FIT_epochs if params_in["auto_aug_epochs"] in ["None", ""] else int(params_in["auto_aug_epochs"])
learning_rate = [float(v) for v in params_in["learning_rate"].split(",")]
if len(learning_rate)==1:
    learning_rate.append(learning_rate[0])
CONFIG = {"model_type": params_in["model_type"],  # VGG16 VGG19 ResNet50 Xception
          "COMPILE_loss": params_in["COMPILE_loss"],

          "tuning_strategy": str(params_in["tuning_strategy"]),  # new
          "early_stop": int(params_in["early_stop"]),  # changed to int
          "max_trials": int(params_in["max_trials"]),  # new
          "step_lr": float(params_in["step_lr"]),  # new
          "step_weight_decay": float(params_in["step_weight_decay"]),  # new

          "COMPILE_optimizer": params_in["COMPILE_optimizer"].split(","),  # changed
          "FIT_batch_size": [int(v) for v in params_in["FIT_batch_size"].split(",")],  # changed
          "learning_rate": learning_rate,  # changed
          "activation_function": params_in["activation_function"].split(','),   # new

          "auto_aug_config": auto_aug_config,
          "COMPILE_metrics": params_in["COMPILE_metrics"],
          "FIT_epochs": FIT_epochs,
          "FIT_shuffle": params_in["FIT_shuffle"] == "True",
          "class_num": class_num,  # 去掉参数
          "norm_size": int(params_in["norm_size"]),
          "model_weights": params_in["model_weights"] == "True",  # 是否有必要
          "random_seed": int(params_in["random_seed"]),  # 新增参数
          # "image_augment": params_in["image_augment"] == "True",
          "weights_dir": pretained_weights,
          "tensorboard_dir": tensorboard_dir,
          "augment_type": params_in["image_augment"],
          "auto_aug_ratio": auto_aug_ratio,
          "auto_aug_epochs": auto_aug_epochs,
          "image_col": image_col,
          "label_col": label_col,
          "work_dir": work_dir,
          "population_size": int(params_in["population_size"]),
          "use_amp": params_in["use_amp"]=="True",
          }

dc.logger.info("CONFIG:\n {}".format(repr(CONFIG)))

step = ImageClassificationEstimator(net_config=CONFIG, input_cols=[image_col], target_cols=[label_col], output_cols=output_cols)
print(df_train.columns)
step.fit(df_train.loc[:,[image_col]], df_train[label_col], image_test=df_test.loc[:,[image_col]], label_test=df_test[label_col])


def remove_file(dir_path, file_pattern):

    from pathlib import Path

    g = Path(dir_path).rglob(file_pattern)
    for f in g:
        if f.is_file():
            print("Removing tmp file... {}".format(f.absolute()))
            f.unlink()

remove_file(tensorboard_dir, "*.h5")

step.persist()

step.model.save(dc.conf.outputs.trained_model)

def filter_float16(df_16):
    for k,v in df_16.dtypes.items():
        if v.name=="float16":
            df_16[k] = df_16[k].astype(float)
    return df_16

prediction = step.predict(df_test)
prediction = filter_float16(prediction)

prediction[label_col] = df_test[label_col].values
prediction.dropna(axis=0, inplace=True)
dc.dataset(dc.conf.outputs.prediction).update(prediction)

# ----------------------shap-------------------------------------
import keras.backend as K
import numpy as np
import shap
from matplotlib import pyplot as plt

def shap_picture(df,back_picture_num,explan_picture_num):
    model = step.model
    df = step.prepocess(df)
    df = df.astype("float32")
    e = shap.GradientExplainer(
        step.model,
        df[explan_picture_num:explan_picture_num+back_picture_num],
        local_smoothing=0 # std dev of smoothing noise
    )
    shap_values,indexes = e.shap_values(df[0:explan_picture_num], ranked_outputs=2)
    return shap_values, df

back_picture_num = int(params_in["back_picture_num"])    
explan_picture_num = int(params_in["explan_picture_num"])  
max_num = df_test.shape[0]
if max_num<back_picture_num+1:
    print("back_picture_num[{}]超过了可用样本数[{}]，修改成[{}]".format(back_picture_num, max_num-1, max_num-1))
    back_picture_num = max_num-1
if max_num<explan_picture_num+back_picture_num:
    print("explan_picture_num[{}]超过了可用样本数[{}]，修改成[{}]".format(explan_picture_num, max_num-back_picture_num, max_num-back_picture_num))
    explan_picture_num = max_num-back_picture_num
if explan_picture_num<=0 or back_picture_num<=0:
    raise Exception("用于生成模型解释的样本不够！")

shap_values, df = shap_picture(df_test, back_picture_num, explan_picture_num)
# plot the explanations
shap.image_plot(shap_values, df[0:explan_picture_num], show=False)
plt.savefig(str(dc.conf.outputs.shap_picture), format='png')
#-------------------hot-------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import tensorflow.keras.backend as K

def draw_cam(original_img_path=None, img_tensor=None, model=None, heatmap_path=None):
    import cv2
    import numpy as np
    import tensorflow.keras.backend as K
    import tensorflow as tf
    original_img = cv2.imread(original_img_path)
    x_g = tf.Variable(img_tensor,dtype='float32')
    with tf.GradientTape() as t:
        y_raw = model(x_g)
        prob = tf.gather(y_raw, tf.argmax(y_raw, axis=1), axis=1)  # 最大可能性类别的预测概率 tf.argmax(xx, axis=1)
    g= t.gradient(prob, x_g)
    pooled_grads = K.mean(g, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, x_g), axis=-1)  # 权重与特征层相乘，512层求和平均

    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    heatmap1 = np.maximum(heatmap1, 0)
    heatmap1 = heatmap1 / np.max(heatmap1)

    heatmap1 = np.uint8(255 * heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    frame_out = cv2.addWeighted(original_img, 0.5, heatmap1, 0.5, 0)
    # frame_out = heatmap1*0.4 + original_img
    cv2.imwrite(heatmap_path, frame_out)

# def hot(df,picture_index):
#     img = step.prepocess(df)
#     img = np.array([img[picture_index]]) #---------取第picture_index个图片---------
#     Predictions = step.model.predict(img)
    
#     if CONFIG['model_type']=='LeNet':
#         step.model.summary()
#         last_conv_layer = step.model.get_layer(index=0)
#     elif CONFIG['model_type']=='VGG16':
#         step.model.get_layer(index=3).summary()
#         last_conv_layer = step.model.get_layer(index=3).get_layer('block5_conv3')
#     elif CONFIG['model_type']=='VGG19':
#         step.model.get_layer(index=3).summary()
#         last_conv_layer = step.model.get_layer(index=3).get_layer('block5_conv4')
#     elif CONFIG['model_type']=='ResNet50':
#         step.model.get_layer(index=3).summary()
#         last_conv_layer = step.model.get_layer(index=3).get_layer('conv5_block3_3_conv')
#     else:
#         step.model.get_layer(index=3).summary()
#         last_conv_layer = step.model.get_layer(index=3).get_layer('block14_sepconv2')
    
#     heatmap_model = models.Model([step.model.inputs], [last_conv_layer.output, step.model.outputs])
    
#     with tf.GradientTape() as gtape:
#         conv_output, Predictions = heatmap_model(img)
#         prob = tf.gather(Predictions[0],tf.argmax(Predictions[0], axis=1), axis=1) # 最大可能性类别的预测概率 tf.argmax(xx, axis=1)
#         grads = gtape.gradient(prob, conv_output)  # 类别与卷积层的梯度 (1,14,14,512)
#         pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重
    
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)  # 权重与特征层相乘，512层求和平均
    
#     heatmap = np.maximum(heatmap, 0)
#     max_heat = np.max(heatmap)
#     if max_heat == 0:
#         max_heat = 1e-10
#     heatmap /= max_heat
    
#     original_img = cv2.imread(list(df['image_path'].values)[picture_index])  #取第一个图片路径
#     heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
#     heatmap1 = np.maximum(heatmap1, 0)
#     heatmap1 = heatmap1 / np.max(heatmap1)
    
#     heatmap1 = np.uint8(255 * heatmap1)
#     heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
#     frame_out = cv2.addWeighted(original_img, 0.5, heatmap1, 0.5, 0)
    
#     return frame_out

picture_index = int(params_in["picture_index"])
img = step.prepocess(df_test)
img = np.array([img[picture_index]]) 

draw_cam(original_img_path=list(df_test['image_path'].values)[picture_index], img_tensor=img, model=step.model, heatmap_path=str(dc.conf.outputs.hot_picture))
# frame_out = hot(df_test,picture_index)
# cv2.imwrite(str(dc.conf.outputs.hot_picture), frame_out)
#---------------------------------------------------------

dc.logger.info("Done!")
