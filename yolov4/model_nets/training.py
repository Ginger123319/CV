import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model_nets.utils.dataloader import yolo_dataset_collate, YoloDataset
from model_nets.utils.dataloader2 import YoloDataset2
from model_nets.nets.yolo_training import YOLOLoss, Generator
from model_nets.nets.yolo4 import YoloBody
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datacanvas.aps import dc


class Train(object):
    # 获取目标检测的类别名称
    def get_classes(self, classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 获取先验框数值
    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # 获取学习率
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # 每一个epoch的训练，验证，保存模型权重的步骤
    def fit_ont_epoch(self, net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, writer):
        total_loss = 0
        val_loss = 0
        start_time = time.time()
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                if targets is None:
                    print(f"Skip batch {iteration}: no label")
                    continue
                with torch.no_grad():
                    if cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = net(images)
                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += float(loss)
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'lr': self.get_lr(self.optimizer),
                                    'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()

        # 将loss写入tensorboard，每个epoch保存一次
        writer.add_scalar('Train_loss', total_loss / (iteration + 1), epoch)
        net.eval()
        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                images_val, targets_val = batch[0], batch[1]
                if targets_val is None:
                    print(f"Skip batch {iteration}: no label")
                    continue
                with torch.no_grad():
                    if cuda:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    else:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    self.optimizer.zero_grad()
                    outputs = net(images_val)
                    losses = []

                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets_val)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += float(loss)

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)
        net.train()
        # # 将loss写入tensorboard，每个epoch保存一次
        writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), epoch)

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ' - loss: %.3f  - val_loss: %.3f' % (
        total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

        if epoch == Epoch - 1:
            torch.save(self.model.state_dict(),
                       self.work_dir + '/middle_dir/logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
                       (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    def __init__(self, num_classes, image_size, val_size, Cosine_lr, mosaic, smooth_label, work_dir, tensorboard_dir,
                 use_tfrecord, use_amp):
        # 参数初始化
        self.num_classes = num_classes  # 目标检测的类别个数
        self.input_shape = (image_size, image_size)  # 图片的尺寸
        self.val_size = val_size
        self.Cosine_lr = Cosine_lr
        self.mosaic = mosaic
        self.smoooth_label = smooth_label
        self.work_dir = work_dir
        self.Cuda = torch.cuda.is_available()  # 检测gpu是否可用
        self.use_tfrecord = use_tfrecord
        self.use_amp = use_amp
        if self.Cuda == False:
            self.use_amp = False
        print("Cuda:", self.Cuda, " use_amp:", self.use_amp, ' use_tfrecord:', self.use_tfrecord)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.annotation_path = self.work_dir + '/middle_dir/data_info/train_info.txt'  # 数据集标注信息的路径
        self.anchors_path = 'script_files/yolo_anchors.txt'
        self.anchors = self.get_anchors(self.anchors_path)
        self.model_path = self.work_dir + "/pre_training_weights/pre_training_weights.pth"  # 预训练权重文件的路径

        self.model = YoloBody(len(self.anchors[0]), self.num_classes)  # 实例化模型网络

        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.Cuda else 'cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished Loading weights!')

        self.net = self.model.train()
        if self.Cuda:
            gpu_num = torch.cuda.device_count()
            device_ids = []
            for i in range(gpu_num):
                device_ids.append(i)
            self.net = torch.nn.DataParallel(self.model, device_ids=device_ids)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        # 建立loss函数
        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(np.reshape(self.anchors, [-1, 2]), self.num_classes,
                                             (self.input_shape[1], self.input_shape[0]), self.smoooth_label, self.Cuda))

        # 将训练集拆分成训练集和验证集
        with open(self.annotation_path) as f:
            self.lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(self.lines)
        np.random.seed(None)
        self.num_val = int(len(self.lines) * self.val_size)
        self.num_train = len(self.lines) - self.num_val

        self.writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=60)
        if self.Cuda:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, self.input_shape[0], self.input_shape[1])).type(
                torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, self.input_shape[0], self.input_shape[1])).type(
                torch.FloatTensor)
        self.writer.add_graph(self.model, (graph_inputs,))

        if self.use_tfrecord:
            # 在训练之前，先做数据增强，并把数据保存在tfrecord中
            from . import data_augment
            from .utils import create_index
            from multiprocessing import Process

            self.train_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord")
            self.train_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord.index")

            self.val_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord")
            self.val_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord.index")

            # 以子进程的方式生成tfrecord文件和index文件
            p = Process(target=data_augment.augment_train_data,
                        args=(self.lines[:self.num_train], self.train_tf_record_file, self.input_shape))
            p.start()
            p.join()
            p.terminate()
            create_index.create_index(self.train_tf_record_file, self.train_tf_record_index)

            p = Process(target=data_augment.augment_val_data,
                        args=(self.lines[self.num_train:], self.val_tf_record_file, self.input_shape))
            p.start()
            p.join()
            p.terminate()
            create_index.create_index(self.val_tf_record_file, self.val_tf_record_index)

    def train(self, lr, freeze_epoch, total_epoch, optimizer, batch_size):

        epoch_size = max(1, self.num_train // batch_size)
        epoch_size_val = self.num_val // batch_size

        if freeze_epoch > 0:
            lr = lr
            Init_Epoch = 0
            Freeze_Epoch = freeze_epoch

            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr, weight_decay=5e-4)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr, weight_decay=5e-4)

            if self.Cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

            if self.use_tfrecord:
                train_dataset = YoloDataset2(self.lines[:self.num_train], self.train_tf_record_file,
                                             self.train_tf_record_index, (self.input_shape[0], self.input_shape[1]),
                                             mosaic=self.mosaic)
                val_dataset = YoloDataset2(self.lines[self.num_train:], self.val_tf_record_file,
                                           self.val_tf_record_index, (self.input_shape[0], self.input_shape[1]),
                                           mosaic=False)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True,
                                 collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True,
                                     collate_fn=yolo_dataset_collate)
            else:
                train_dataset = YoloDataset(self.lines[:self.num_train], (self.input_shape[0], self.input_shape[1]),
                                            mosaic=self.mosaic)
                val_dataset = YoloDataset(self.lines[self.num_train:], (self.input_shape[0], self.input_shape[1]),
                                          mosaic=False)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)

            #  冻结主干网络权重进行训练
            for param in self.model.backbone.parameters():
                param.requires_grad = False

            for epoch in range(Init_Epoch, Freeze_Epoch):
                self.fit_ont_epoch(self.net, self.yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                   Freeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()

        if total_epoch - freeze_epoch > 0:
            lr = 0.0001
            Freeze_Epoch = freeze_epoch
            Unfreeze_Epoch = total_epoch

            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr, weight_decay=5e-4)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr, weight_decay=5e-4)

            if self.Cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

            if self.use_tfrecord:
                train_dataset = YoloDataset2(self.lines[:self.num_train], self.train_tf_record_file,
                                             self.train_tf_record_index, (self.input_shape[0], self.input_shape[1]),
                                             mosaic=self.mosaic)
                val_dataset = YoloDataset2(self.lines[self.num_train:], self.val_tf_record_file,
                                           self.val_tf_record_index, (self.input_shape[0], self.input_shape[1]),
                                           mosaic=False)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True,
                                 collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True,
                                     collate_fn=yolo_dataset_collate)
            else:
                train_dataset = YoloDataset(self.lines[:self.num_train], (self.input_shape[0], self.input_shape[1]),
                                            mosaic=self.mosaic)
                val_dataset = YoloDataset(self.lines[self.num_train:], (self.input_shape[0], self.input_shape[1]),
                                          mosaic=False)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)

            #  结冻主干网络权重进行训练
            for param in self.model.backbone.parameters():
                param.requires_grad = True

            for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
                self.fit_ont_epoch(self.net, self.yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                   Unfreeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()
