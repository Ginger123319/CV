#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import cv2
from mosaic_all import get_mosaic_data
from aug_methods import func_dict
from aug_methods import *
from utils import read_rgb_img, get_image_list, get_methods, get_labels_selected, create_annotation, \
    get_image_list_mix_mosaic
import random
from datacanvas.aps import dc
from dc_model_repo import model_repo_client
from pathlib import Path
import json
import datetime
import uuid
import copy
import shutil


def augmentation(label_type, is_augment_no_labeled_data, augment_methods, labels_selected, merge_strategy, image_col,
                 label_col, partition_names, input_data_path, output_data_path, augment_scope):
    if label_type == "103":
        augment_scope = augment_scope
    else:
        augment_scope = None
    is_augment_no_labeled_data = True if is_augment_no_labeled_data.lower() == "true" else False

    # 处理输出路径,不存在就创建
    if not os.path.exists(output_data_path):
        Path(output_data_path).mkdir(parents=True, exist_ok=True)

    # 可能为空列表
    labels_selected_list = get_labels_selected(labels_selected)

    # 获取用户选择的增强方法
    methods_list = get_methods(augment_methods)

    # 暂定传入一个包含分片名称，分割的字符串，先将其处理为列表
    partition_names = partition_names.split(',')

    for name in partition_names:

        # 处理图片和json文件保存路径

        image_path = Path(output_data_path, name, 'image')
        if image_path.exists():
            shutil.rmtree(image_path)
        image_path.mkdir(parents=True, exist_ok=True)
        image_relative_path = Path(name, 'image')
        json_path = Path(output_data_path, name, 'annotation.json')

        # 读入数据
        # label_all将未标注的数据读出来；partition_names筛选对应的数据集分片
        ds = dc.dataset(input_data_path).read(labeled_all=True, selected_part=[name])

        # 图片标签列表，每一个元素是一个包含图片路径和标签的列表
        image_list, class_name, file_name = get_image_list(ds, image_col, label_col, label_type, labels_selected_list,
                                                           is_augment_no_labeled_data)

        current_time = datetime.datetime.now()
        annotation = {
            "info": {
                "description": "Fastlabel_CV数据增强",
                "url": "",
                "version": "1.0",
                "year": str(current_time)[:4],
                "contributor": "",
                "date_created": str(current_time)
            },
            "licenses": [
                {
                    "url": "",
                    "id": 0,
                    "name": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": -1,
                    "name": "__ignore__"
                },
                {
                    "id": 0,
                    "name": "_background_"
                },
            ]
        }

        # 写入类别名称
        for cls_name, cls_id in class_name.items():
            annotation['categories'].append({
                'id': cls_id,
                'name': cls_name
            })

        # 为mix和mosaic提供的列表
        for method in methods_list:
            print(method)

            if method == "mix":
                image_list_changed = []
                for image_label, filename in zip(image_list, file_name):

                    label = image_label[1]
                    image = image_label[0]
                    filename = filename.split('/')[-1]

                    # print("没有标签的图片不支持混合操作")
                    if len(label) == 0:
                        continue
                    # 图像分类单标签，需要同一个类别的图片做混合和马赛克操作
                    image_list_mix_mosaic = get_image_list_mix_mosaic(image, label, image_list, label_type)

                    if len(image_list_mix_mosaic) >= 1:
                        mix_list = random.sample(image_list_mix_mosaic, 1)
                    else:
                        print("图片数目：{} 图片过少无法进行混合增强！".format(len(image_list_mix_mosaic)))
                        continue
                    mix_list.append(image_label)
                    mix_list = copy.deepcopy(mix_list)
                    mix_img, mix_label = func_dict[method](mix_list, label_type, augment_scope)

                    if merge_strategy == "parallel":
                        filename = 'mix', str(uuid.uuid1())[2:8], filename
                        filename = '-'.join(filename)
                        filepath = os.path.join(image_path, filename)
                        relative_path = os.path.join(image_relative_path, filename)
                        mix_img = cv2.cvtColor(mix_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(filepath, mix_img)
                        create_annotation(annotation, relative_path, filename, mix_img, mix_label, class_name,
                                          label_type)
                    else:
                        image_list_changed.append([mix_img, mix_label])
                if merge_strategy == "serial":
                    image_list = copy.deepcopy(image_list_changed)

            elif method == "mosaic":
                image_list_changed = []
                for image_label, filename in zip(image_list, file_name):

                    label = image_label[1]
                    image = image_label[0]
                    filename = filename.split('/')[-1]

                    # print("没有标签的图片不支持马赛克操作")
                    if len(label) == 0:
                        continue

                    # 图像分类单标签，需要同一个类别的图片做混合和马赛克操作
                    image_list_mix_mosaic = get_image_list_mix_mosaic(image, label, image_list, label_type)

                    if len(image_list_mix_mosaic) >= 3:
                        mosaic_list = random.sample(image_list_mix_mosaic, 3)
                    else:
                        print("图片数目：{} 图片过少无法进行马赛克增强！".format(len(image_list_mix_mosaic)))
                        continue
                    mosaic_list.append(image_label)
                    mosaic_list = copy.deepcopy(mosaic_list)
                    mosaic_img, mosaic_label = func_dict[method](mosaic_list, label_type, augment_scope)
                    if merge_strategy == "parallel":
                        filename = 'mosaic', str(uuid.uuid1())[2:8], filename
                        filename = '-'.join(filename)
                        filepath = os.path.join(image_path, filename)
                        relative_path = os.path.join(image_relative_path, filename)
                        mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(filepath, mosaic_img)
                        create_annotation(annotation, relative_path, filename, mosaic_img, mosaic_label, class_name,
                                          label_type)
                    else:
                        image_list_changed.append([mosaic_img, mosaic_label])
                if merge_strategy == "serial":
                    image_list = copy.deepcopy(image_list_changed)

            else:
                if method in func_dict.keys():
                    image_list_changed = []
                    for image_label, filename in zip(image_list, file_name):
                        filename = filename.split('/')[-1]
                        image_label = copy.deepcopy(image_label)
                        enhance_img, enhance_label = func_dict[method](image_label, label_type, augment_scope)
                        if merge_strategy == "parallel":
                            filename = str(method), str(uuid.uuid1())[2:8], filename
                            filename = '-'.join(filename)
                            filepath = os.path.join(image_path, filename)
                            relative_path = os.path.join(image_relative_path, filename)
                            enhance_img = cv2.cvtColor(enhance_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(filepath, enhance_img)
                            create_annotation(annotation, relative_path, filename, enhance_img, enhance_label,
                                              class_name, label_type)
                        else:
                            image_list_changed.append([enhance_img, enhance_label])

                    if merge_strategy == "serial":
                        image_list = copy.deepcopy(image_list_changed)
                else:
                    raise Exception("不支持这种增强方法：{}".format(method))

        if merge_strategy == "serial":
            for image_label, filename in zip(image_list, file_name):
                label = image_label[1]
                image = image_label[0]
                filename = filename.split('/')[-1]

                filename = 'serial', str(uuid.uuid1())[2:8], filename
                filename = '-'.join(filename)
                filepath = os.path.join(image_path, filename)
                relative_path = os.path.join(image_relative_path, filename)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(filepath, img)
                create_annotation(annotation, relative_path, filename, img, label, class_name, label_type)

        json_data = json.dumps(annotation)
        with open(json_path, 'w') as f:
            f.write(json_data)
