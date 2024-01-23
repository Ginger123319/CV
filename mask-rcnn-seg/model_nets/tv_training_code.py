# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import cv2
import torch
import io
import re
import json
import shutil
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import struct
import typing
import tfrecord

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tensorboardX import SummaryWriter
from tfrecord.torch.dataset import MultiTFRecordDataset

from .detection.engine import train_one_epoch, evaluate
from .detection import utils
from .detection import transforms as T
from . import cv2_util


class COCODataset(object):

    def __init__(self, transforms, dataset, class_name, dataset_len):
        self.transforms = transforms
        self.coco = dataset
        self.class_name = class_name
        self.dataset_len = dataset_len
        self.ids = list(self.coco.imgToAnns.keys())

        self.class_map = dict()
        for k, v in self.coco.cats.items():
            self.class_map[k] = v['name']

    def parse_label(self, target):
        labels = []
        boxes = []
        for obj in target:
            bbox = obj['bbox']
            boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])
            labels.append(self.class_name.index(self.class_map[obj['category_id']]))
        return labels, boxes

    def __getitem__(self, index):
        image_id = self.ids[index]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        img_path = self.coco.loadImgs(image_id)[0]["file_full_path"]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        masks = np.array([self.coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
        masks = masks.reshape((-1, height, width))
        labels, boxes = self.parse_label(target)

        num_objs = len(masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.dataset_len


class InsSegDataset(object):

    def __init__(self, transforms, dataset, class_name, dataset_len):
        self.transforms = transforms
        self.dataset = dataset
        self.class_name = class_name
        self.dataset_len = dataset_len

    def parse_label(self, xml_path):
        label_root = ET.parse(xml_path).getroot()

        labels = []
        boxes = []
        for obj in label_root.iter('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue

            cls = obj.find('name').text
            labels.append(self.class_name.index(cls))

            xmin = int(obj.find("bndbox").find('xmin').text)
            ymin = int(obj.find("bndbox").find('ymin').text)
            xmax = int(obj.find("bndbox").find('xmax').text)
            ymax = int(obj.find("bndbox").find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return labels, boxes

    def __getitem__(self, index):
        img_path = self.dataset.get_item(index).data
        mask_path = self.dataset.get_item(index).mask
        label_path = self.dataset.get_item(index).label

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        labels, boxes = self.parse_label(label_path)

        # 删除mask中的白色部分
        mask[mask == 255] = 0
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.dataset_len


def get_model_instance_segmentation(num_classes, device, pretrained_pth, trainable_backbone_layers):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, trainable_backbone_layers=None, num_classes=num_classes)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_pth, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    def freeze(resnet_model, trainable_layers):
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        if trainable_layers == 5:
                layers_to_train.append("bn1")
        print(f"Layers to train: {layers_to_train}")
        for name, parameter in resnet_model.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                print(f"Freeze: {name}")
                parameter.requires_grad_(False)
    freeze(model.backbone.body, trainable_backbone_layers)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256


def training(class_name, trainable_backbone_layers, total_epoch, lr, batch_size, optimizer, weight_decay, pretrained_pth, device, model_dir, tensorboard_dir,
             work_dir, train_data_len, test_data_len, dataset_type, train_dataset, val_dataset):
    # 实例化数据集的DataSet
    if dataset_type == "VOC":
        dataset = InsSegDataset(get_transform(train=True), train_dataset, class_name, train_data_len)
        dataset_test = InsSegDataset(get_transform(train=False), val_dataset, class_name, test_data_len)
    elif dataset_type == "COCO":
        dataset = COCODataset(get_transform(train=True), train_dataset.data, class_name, train_data_len)
        dataset_test = COCODataset(get_transform(train=False), val_dataset.data, class_name, test_data_len)
    else:
        raise Exception("do not support this data type!")

    # 构建DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, num_workers=0, collate_fn=utils.collate_fn)
    # 实例化模型类
    model = get_model_instance_segmentation(len(class_name), device, pretrained_pth, trainable_backbone_layers).to(device)
    # 选择优化器
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # 实例化SummaryWriter，保存模型训练的日志
    writer = SummaryWriter(log_dir=work_dir, flush_secs=60)
    resnet = torchvision.models.resnet50()
    graph_inputs_ = torch.from_numpy(np.random.rand(1, 3, 7, 7)).to(torch.float)
    with torch.no_grad():
        writer.add_graph(resnet, (graph_inputs_,))
    print("nets graph has been writen!")
    # 开始训练
    train_log_file = open(os.path.join(work_dir, 'train.txt'), 'a')  # 创建train log file来保存训练时的train和val的损失
    loss_before = 1000000
    for epoch in range(total_epoch):
        losses_value = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        train_loss = torch.tensor(round(losses_value, 3)).to(device)
        # 更新学习率
        lr_scheduler.step()
        # 在测试集上进行评估
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        bbox_miou = coco_evaluator.coco_eval['bbox'].stats.max()
        segm_miou = coco_evaluator.coco_eval['segm'].stats.max()
        total_miou = bbox_miou + segm_miou
        total_miou = torch.tensor(round(total_miou, 3)).to(device)
        val_loss_value = 1 - coco_evaluator.coco_eval['bbox'].eval['scores'].max()
        val_loss = torch.tensor(round(val_loss_value, 3)).to(device)

        print(f"epoch: {epoch + 1}, total train loss: {round(losses_value, 4)}, total val loss: {round(val_loss_value, 4)}")
        train_log = 'Epoch:' + str(epoch + 1) + '/' + str(total_epoch) + ' - loss: %.3f  - val_loss: %.3f' % (losses_value, val_loss_value)
        train_log_file.write(train_log)
        train_log_file.write('\n')

        # 记录每个epoch的train_loss, val_loss, total_MIOU
        writer.add_scalar('EvalTotalMiou', total_miou, epoch)
        writer.add_scalar('TrainLoss', train_loss, epoch)
        writer.add_scalar('ValLoss', val_loss, epoch)
        if losses_value < loss_before:
            loss_before = losses_value
            torch.save(model.state_dict(), os.path.join(work_dir, 'model.pth'))
    train_log_file.close()

    performance = [{"type": "metrics", "name": "metrics", "data": {"bbox_miou": float(bbox_miou), "segm_miou": float(segm_miou), "total_miou": float(total_miou)}}]
    return loss_before, performance


def train_func(class_name, trainable_backbone_layers, step_lr, step_weight_decay, total_epoch, max_trials, early_stop, tuning_strategy, lr,
               batch_size, optimizer, weight_decay, activation_function, pretrained_pth, device, model_dir, tensorboard_dir,
               performance_path, work_dir, train_data_len, test_data_len, dataset_type, train_dataset, val_dataset):
    from hypernets.utils.param_tuning import search_params
    from hypernets.core.search_space import Choice, Real
    from hypernets.core import EarlyStoppingCallback

    print("Hyperparameter search space are lr:%s, batch_size:%s, optimizer:%s, activation_function:%s, weight_decay:%s." % (lr, batch_size, optimizer, activation_function, weight_decay))
    if len(lr) == 1:
        lr = float(lr[0])
    else:
        tmp_a = float(lr[1]) - float(lr[0])
        if tmp_a/step_weight_decay<10:
            print(f"Force step_lr from {step_lr} to {tmp_a/10}")
            step_lr = tmp_a/10
        
        lr = Real(float(lr[0]), float(lr[1]), step=step_lr)
    if len(batch_size) == 1:
        batch_size = int(batch_size[0])
    else:
        batch_size = Choice([int(i) for i in batch_size])
    if len(optimizer) == 1:
        optimizer = optimizer[0]
    else:
        optimizer = Choice(optimizer)
    if len(activation_function) == 1:
        activation_function = activation_function[0]
    else:
        activation_function = Choice(activation_function)
    if len(weight_decay) == 1:
        weight_decay = float(weight_decay[0])
    else:
        tmp_a = float(weight_decay[1]) - float(weight_decay[0])
        if tmp_a/step_weight_decay<10:
            print(f"Force step_weight_decay from {step_weight_decay} to {tmp_a/10}")
            step_weight_decay = tmp_a/10
        weight_decay = Real(float(weight_decay[0]), float(weight_decay[1]), step=step_weight_decay)
    best_params_ = {'lr': lr, 'batch_size': batch_size, 'optimizer': optimizer, 'activation_function': activation_function, 'weight_decay': weight_decay}

    def train_function(lr=lr, batch_size=batch_size, optimizer=optimizer, activation_function=activation_function, weight_decay=weight_decay):

        loss_before, performance = training(class_name, trainable_backbone_layers, total_epoch, lr, batch_size, optimizer, weight_decay, pretrained_pth, device, model_dir,
                                            tensorboard_dir, work_dir, train_data_len, test_data_len, dataset_type, train_dataset, val_dataset)

        # 当一个trial结束后，更改model.pth文件的名称，更改tensorboard文件的名称，创建performance文件
        shutil.move(os.path.join(work_dir, 'train.txt'), os.path.join(work_dir, f'{round(loss_before, 5)}.txt'))
        shutil.move(os.path.join(work_dir, 'model.pth'), os.path.join(work_dir, f'{round(loss_before, 5)}.pth'))
        for i in os.listdir(work_dir):
            if i.startswith('events.'):
                shutil.move(os.path.join(work_dir, i), os.path.join(work_dir, f'{round(loss_before, 5)}.tfevents'))
        with open(os.path.join(work_dir, f'{round(loss_before, 5)}.json'), 'w') as f:
            json.dump(performance, f)
        print(f"[Trial] reward={round(loss_before, 4)} lr={lr} batch_size={batch_size} optimizer={optimizer} activation_function={activation_function} weight_decay={weight_decay}")
        return loss_before

    es = EarlyStoppingCallback(max_no_improvement_trials=early_stop, mode='min')
    print('Start trial.')
    history = search_params(func=train_function, searcher=tuning_strategy, max_trials=max_trials, optimize_direction='min', callbacks=[es])
    best = history.get_best()
    ps = best.space_sample.get_assigned_params()
    best_params = {p.alias.split('.')[-1]: p.value for p in ps}
    best_params_.update(best_params)

    # 当所有的trial结束后，保存表现最好的trail对应的模型权重文件，tensorboard文件和json文件
    shutil.copyfile(os.path.join(work_dir, f'{round(best.reward, 5)}.pth'), os.path.join(model_dir, 'model.pth'))
    shutil.copyfile(os.path.join(work_dir, f'{round(best.reward, 5)}.tfevents'), os.path.join(tensorboard_dir, 'events.out.tfevents'))
    with open(os.path.join(work_dir, f'{round(best.reward, 5)}.json'), 'r') as f:
        performance = json.load(f)
    with open(performance_path, 'w') as f:
        json.dump(performance, f)
    print("best trial train logs:")
    with open(os.path.join(work_dir, f'{round(best.reward, 5)}.txt'), 'r') as f:
        contents = f.readlines()
        for content in contents:
            print(content)
    print("best_params:", best_params_)
    print('Finished trial.')

    return best_params_


