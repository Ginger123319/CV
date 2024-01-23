from model_nets.utils.trainer import FasterRCNNTrainer
from model_nets.nets.frcnn import FasterRCNN
from model_nets.utils.dataloader2 import FRCNNDataset2, frcnn_dataset_collate
from model_nets.utils.dataloader import FRCNNDataset
import os 
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class Train(object):
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_ont_epoch(self, net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, writer):
        total_loss = 0
        rpn_loc_loss = 0
        rpn_cls_loss = 0
        roi_loc_loss = 0
        roi_cls_loss = 0
        val_toal_loss = 0
        iteration = 0
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                imgs, boxes, labels = batch[0], batch[1], batch[2]

                with torch.no_grad():
                    if cuda:
                        imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                        boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                        labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
                    else:
                        imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                        boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)) for box in boxes]
                        labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)) for label in labels]
                        
                self.optimizer.zero_grad()
                losses = net(imgs, boxes, labels, 1)
                rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
                losses.total_loss.backward()
                self.optimizer.step()                
                
                total_loss += float(total)
                rpn_loc_loss += float(rpn_loc)
                rpn_cls_loss += float(rpn_cls)
                roi_loc_loss += float(roi_loc)
                roi_cls_loss += float(roi_cls)

                pbar.set_postfix(**{'total': total_loss / (iteration + 1),
                                    'rpn_loc': rpn_loc_loss / (iteration + 1),
                                    'rpn_cls': rpn_cls_loss / (iteration + 1),
                                    'roi_loc': roi_loc_loss / (iteration + 1),
                                    'roi_cls': roi_cls_loss / (iteration + 1),
                                    'lr': self.get_lr(self.optimizer)})
                pbar.update(1)
                
        if writer:
            # 将loss写入tensorboard，每个epoch保存一次
            writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
        
        print('Start Validating...')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                imgs, boxes, labels = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    if cuda:
                        imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                        boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                        labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
                    else:
                        imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                        boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)) for box in boxes]
                        labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)) for label in labels]

                    self.train_util.optimizer.zero_grad()
                    losses = self.train_util.forward(imgs, boxes, labels, 1)
                    _, _, _, _, val_total = losses
                    val_toal_loss += float(val_total)
                pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1)})
                pbar.update(1)
                
        if writer:                
            # 将loss写入tensorboard，每个epoch保存一次
            writer.add_scalar('Val_loss',val_toal_loss/(epoch_size_val+1), epoch)    
        
        print('Finish Validating!')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ' - loss: %.3f  - val_loss: %.3f' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))
        
        if epoch == Epoch - 1 and int(os.environ.get("RANK")) == 0:
            torch.save(self.model.state_dict(), self.work_dir+'/middle_dir_mask_dis/logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))
 
        
    def __init__(self, num_classes, image_size, val_size, work_dir, tensorboard_dir, dist, use_tfrecord, use_amp, mosaic=False):
        # 参数初始化
        self.mosaic = mosaic
        self.num_classes = num_classes # 目标检测的类别个数
        self.image_shape = [image_size, image_size, 3] # 图片的尺寸
        self.val_size = val_size
        self.work_dir = work_dir
        self.Cuda = torch.cuda.is_available() # 检测gpu是否可用
        self.use_tfrecord = use_tfrecord
        self.use_amp = use_amp
        if self.Cuda == False:
            self.use_amp = False
        print("Cuda:", self.Cuda, " use_amp:", self.use_amp, ' use_tfrecord:', self.use_tfrecord)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) 
        
        self.annotation_path = work_dir+'/middle_dir_mask_dis/data_info/train_info.txt' # 数据集标注信息的路径
        self.model_path = work_dir+"/pre_training_weights_mask/pre_training_weights.pth" # 预训练权重文件的路径
        BACKBONE = "resnet50" # 主干网络的类型
        self.model = FasterRCNN(self.num_classes, backbone=BACKBONE)
        
        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.Cuda else 'cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished Loading weights!')
        
        self.net = self.model.train()
        self.optimizer = optim.Adam(self.net.parameters(), 0.0001, weight_decay=5e-4)
        
        if self.Cuda:
            gpu_num = torch.cuda.device_count()
            device_ids = []
            for i in range(gpu_num):
                device_ids.append(i)               
            torch.cuda.manual_seed(1)
            self.model = self.model.to(device)
            self.train_util = FasterRCNNTrainer(self.model, self.optimizer)
            self.train_util = self.train_util.to(device)
            self.net = torch.nn.parallel.DistributedDataParallel(self.train_util, find_unused_parameters=True, device_ids=device_ids)
            
            # ============= with powersgb ===========
            import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
            state = powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=2,
                use_error_feedback=True, warm_start=True, random_seed=0
            )
            self.net.register_comm_hook(state, powerSGD.powerSGD_hook)
        else:
            torch.manual_seed(1)
            self.train_util = FasterRCNNTrainer(self.model, self.optimizer)
            self.net = torch.nn.parallel.DistributedDataParallel(self.train_util, find_unused_parameters=True)

        # 将训练集拆分成训练集和验证集
        with open(self.annotation_path) as f:
            self.lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(self.lines)
        np.random.seed(None)
        self.num_val = int(len(self.lines) * self.val_size)
        self.num_train = len(self.lines) - self.num_val
        
        if int(os.environ.get("RANK")) == 0:
            self.writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=60)  
        else:
            self.writer = False

        if self.use_tfrecord:
            # 在训练之前，先做数据增强，并把数据保存在tfrecord中  
            self.train_tf_record_file = os.path.join(self.work_dir, "middle_dir_mask_dis/tfrecord_data/train.tfrecord")
            self.train_tf_record_index = os.path.join(self.work_dir, "middle_dir_mask_dis/tfrecord_data/train.tfrecord.index")
            self.val_tf_record_file = os.path.join(self.work_dir, "middle_dir_mask_dis/tfrecord_data/val.tfrecord")
            self.val_tf_record_index = os.path.join(self.work_dir, "middle_dir_mask_dis/tfrecord_data/val.tfrecord.index")  
            
            if int(os.environ.get("RANK")) == 0:
                from . import data_augment
                from .utils import create_index
                from multiprocessing import Process
                
                # 以子进程的方式生成tfrecord文件和index文件
                p = Process(target=data_augment.augment_train_data, args=(self.lines[:self.num_train], self.train_tf_record_file))
                p.start()
                p.join()
                p.terminate()
                create_index.create_index(self.train_tf_record_file, self.train_tf_record_index)
                
                p = Process(target=data_augment.augment_val_data, args=(self.lines[self.num_train:], self.val_tf_record_file))
                p.start()
                p.join()
                p.terminate()
                create_index.create_index(self.val_tf_record_file, self.val_tf_record_index)   
            
        dist.barrier() 
        

    def train(self, lr, freeze_epoch, total_epoch, optimizer, batch_size):
        
        global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) * batch_size
        epoch_size = max(1, self.num_train//global_batch_size)
        epoch_size_val = self.num_val//batch_size
            
        if freeze_epoch > 0:
            lr = lr
            Init_Epoch = 0
            Freeze_Epoch = freeze_epoch
            
            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr, weight_decay=5e-4)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr, weight_decay=5e-4)
                
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
            
            if self.use_tfrecord:
                train_dataset = FRCNNDataset2(self.lines[:self.num_train], self.train_tf_record_file, self.train_tf_record_index, (self.image_shape[0], self.image_shape[1]), mosaic=self.mosaic)
                val_dataset = FRCNNDataset2(self.lines[self.num_train:], self.val_tf_record_file, self.val_tf_record_index, (self.image_shape[0], self.image_shape[1]))
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
            else:
                train_dataset = FRCNNDataset(self.lines[:self.num_train], (self.image_shape[0], self.image_shape[1]), mosaic=self.mosaic)
                val_dataset = FRCNNDataset(self.lines[self.num_train:], (self.image_shape[0], self.image_shape[1]))
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
                
            #  冻结主干网络权重进行训练
            for param in self.model.extractor.parameters():
                param.requires_grad = False
            self.model.freeze_bn()

            for epoch in range(Init_Epoch, Freeze_Epoch):
                self.fit_ont_epoch(self.net, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()

        if total_epoch > freeze_epoch:
            lr = 1e-5
            Freeze_Epoch = freeze_epoch
            Unfreeze_Epoch = total_epoch

            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr, weight_decay=5e-4)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr, weight_decay=5e-4)
                
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

            if self.use_tfrecord:
                train_dataset = FRCNNDataset2(self.lines[:self.num_train], self.train_tf_record_file, self.train_tf_record_index, (self.image_shape[0], self.image_shape[1]))
                val_dataset = FRCNNDataset2(self.lines[self.num_train:], self.val_tf_record_file, self.val_tf_record_index, (self.image_shape[0], self.image_shape[1]))
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
            else:
                train_dataset = FRCNNDataset(self.lines[:self.num_train], (self.image_shape[0], self.image_shape[1]))
                val_dataset = FRCNNDataset(self.lines[self.num_train:], (self.image_shape[0], self.image_shape[1]))
                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)

            #  结冻主干网络权重进行训练
            for param in self.model.extractor.parameters():
                param.requires_grad = True
            self.model.freeze_bn()

            for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
                self.fit_ont_epoch(self.net, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()

