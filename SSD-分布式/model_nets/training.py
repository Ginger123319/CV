import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_nets.nets.ssd import get_ssd
from model_nets.nets.ssd_training import MultiBoxLoss
from model_nets.utils.config import Config
from model_nets.utils.dataloader2 import SSDDataset2, ssd_dataset_collate
from model_nets.utils.dataloader import SSDDataset
from tensorboardX import SummaryWriter
import os

warnings.filterwarnings("ignore")


class Train(object):
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_one_epoch(self, net, criterion, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, writer):
        loc_loss = 0
        conf_loss = 0
        loc_loss_val = 0
        conf_loss_val = 0

        net.train()
        print('Start Train')
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=3) as pbar:
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
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = net(images)
                    loss_l, loss_c = criterion(out, targets)
                    loss = loss_l + loss_c
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loc_loss += float(loss_l.item())
                conf_loss += float(loss_c.item())

                pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                    'conf_loss': conf_loss / (iteration + 1),
                                    'lr': self.get_lr(self.optimizer)})
                pbar.update(1)

        net.eval()
        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                images, targets = batch[0], batch[1]
                if targets is None:
                    print(f"Skip batch {iteration}: no label")
                    continue

                with torch.no_grad():
                    if cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

                    out = net(images)
                    self.optimizer.zero_grad()
                    loss_l, loss_c = criterion(out, targets)

                    loc_loss_val += float(loss_l.item())
                    conf_loss_val += float(loss_c.item())

                    pbar.set_postfix(**{'loc_loss': loc_loss_val / (iteration + 1),
                                        'conf_loss': conf_loss_val / (iteration + 1),
                                        'lr': self.get_lr(self.optimizer)})
                    pbar.update(1)

        total_loss = loc_loss + conf_loss
        val_loss = loc_loss_val + conf_loss_val
        if writer:
            writer.add_scalar('Train_loss', total_loss / (epoch_size + 1), epoch)
            writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), epoch)

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ' - loss: %.3f  - val_loss: %.3f' % (
        total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
        # print('Saving state, iter:', str(epoch+1))
        if epoch == Epoch - 1 and int(os.environ.get("RANK")) == 0:
            torch.save(self.model.state_dict(),
                       self.work_dir + '/middle_dir_ssd_dis/logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
                       (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    def __init__(self, num_classes, image_size, val_size, work_dir, tensorboard_dir, dist, use_tfrecord, use_amp,
                 mosaic=True):
        self.mosaic = mosaic
        Config["min_dim"] = image_size  # 图片的尺寸
        Config["num_classes"] = num_classes + 1
        self.Cuda = torch.cuda.is_available()  # 检测gpu是否可用
        self.use_tfrecord = use_tfrecord
        self.use_amp = use_amp
        if self.Cuda == False:
            self.use_amp = False
        print("Cuda:", self.Cuda, " use_amp:", self.use_amp, ' use_tfrecord:', self.use_tfrecord)

        self.work_dir = work_dir
        self.annotation_path = self.work_dir + '/middle_dir_ssd_dis/data_info/train_info.txt'
        self.model_path = self.work_dir + "/pre_training_weights_ssd/pre_training_weights.pth"

        self.model = get_ssd("train", Config["num_classes"])

        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.Cuda else 'cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished Loading weights!')

        with open(self.annotation_path) as f:
            self.lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(self.lines)
        np.random.seed(None)
        self.num_val = int(len(self.lines) * val_size)
        self.num_train = len(self.lines) - self.num_val

        self.criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, self.Cuda)

        self.net = self.model.train()
        if self.Cuda:
            gpu_num = torch.cuda.device_count()
            device_ids = []
            for i in range(gpu_num):
                device_ids.append(i)
            torch.cuda.manual_seed(1)
            self.model = self.model.to(device)
            self.net = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True,
                                                                 device_ids=device_ids)

            # ============= with powersgb ===========
            import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
            state = powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=2,
                orthogonalization_epsilon=1e-8,
                use_error_feedback=True, warm_start=True, random_seed=0
            )
            self.net.register_comm_hook(state, powerSGD.powerSGD_hook)
        else:
            torch.manual_seed(1)
            self.net = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

        if int(os.environ.get("RANK")) == 0:
            self.writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=60)
            if self.Cuda:
                graph_inputs = torch.from_numpy(np.random.rand(1, 3, Config["min_dim"], Config["min_dim"])).type(
                    torch.FloatTensor).cuda()
            else:
                graph_inputs = torch.from_numpy(np.random.rand(1, 3, Config["min_dim"], Config["min_dim"])).type(
                    torch.FloatTensor)
            self.writer.add_graph(self.model, (graph_inputs,))
        else:
            self.writer = False

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.use_tfrecord:
            # 在训练之前，先做数据增强，并把数据保存在tfrecord中  
            self.train_tf_record_file = os.path.join(self.work_dir, "middle_dir_ssd_dis/tfrecord_data/train.tfrecord")
            self.train_tf_record_index = os.path.join(self.work_dir,
                                                      "middle_dir_ssd_dis/tfrecord_data/train.tfrecord.index")
            self.val_tf_record_file = os.path.join(self.work_dir, "middle_dir_ssd_dis/tfrecord_data/val.tfrecord")
            self.val_tf_record_index = os.path.join(self.work_dir,
                                                    "middle_dir_ssd_dis/tfrecord_data/val.tfrecord.index")

            if int(os.environ.get("RANK")) == 0:
                from . import data_augment
                from .utils import create_index
                from multiprocessing import Process

                # 以子进程的方式生成tfrecord文件和index文件
                p = Process(target=data_augment.augment_train_data,
                            args=(self.lines[:self.num_train], self.train_tf_record_file))
                p.start()
                p.join()
                p.terminate()
                create_index.create_index(self.train_tf_record_file, self.train_tf_record_index)

                p = Process(target=data_augment.augment_val_data,
                            args=(self.lines[self.num_train:], self.val_tf_record_file))
                p.start()
                p.join()
                p.terminate()
                create_index.create_index(self.val_tf_record_file, self.val_tf_record_index)

        dist.barrier()

    def train(self, lr, freeze_epoch, total_epoch, optimizer, batch_size):

        global_batch_size = int(os.environ.get("WORLD_SIZE", 1)) * batch_size
        epoch_size = max(1, self.num_train // global_batch_size)
        epoch_size_val = self.num_val // batch_size

        if freeze_epoch > 0:
            lr = lr
            Batch_size = batch_size
            Init_Epoch = 0
            Freeze_Epoch = freeze_epoch

            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.92)

            if self.use_tfrecord:
                train_dataset = SSDDataset2(self.lines[:self.num_train], self.train_tf_record_file,
                                            self.train_tf_record_index, (Config["min_dim"], Config["min_dim"]),
                                            mosaic=self.mosaic)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size, pin_memory=True, drop_last=False,
                                 collate_fn=ssd_dataset_collate)
                val_dataset = SSDDataset2(self.lines[self.num_train:], self.val_tf_record_file,
                                          self.val_tf_record_index, (Config["min_dim"], Config["min_dim"]))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, pin_memory=True, drop_last=False,
                                     collate_fn=ssd_dataset_collate)
            else:
                train_dataset = SSDDataset(self.lines[:self.num_train], (Config["min_dim"], Config["min_dim"]), True,
                                           mosaic=self.mosaic)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
                val_dataset = SSDDataset(self.lines[self.num_train:], (Config["min_dim"], Config["min_dim"]), False)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=ssd_dataset_collate)

            for param in self.model.vgg.parameters():
                param.requires_grad = False

            for epoch in range(Init_Epoch, Freeze_Epoch):
                self.fit_one_epoch(self.net, self.criterion, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                   Freeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()

        if total_epoch > freeze_epoch:
            lr = 1e-4
            Batch_size = batch_size
            Freeze_Epoch = freeze_epoch
            Unfreeze_Epoch = total_epoch

            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.92)

            if self.use_tfrecord:
                train_dataset = SSDDataset2(self.lines[:self.num_train], self.train_tf_record_file,
                                            self.train_tf_record_index, (Config["min_dim"], Config["min_dim"]))
                gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size, pin_memory=True, drop_last=False,
                                 collate_fn=ssd_dataset_collate)
                val_dataset = SSDDataset2(self.lines[self.num_train:], self.val_tf_record_file,
                                          self.val_tf_record_index, (Config["min_dim"], Config["min_dim"]))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, pin_memory=True, drop_last=False,
                                     collate_fn=ssd_dataset_collate)
            else:
                train_dataset = SSDDataset(self.lines[:self.num_train], (Config["min_dim"], Config["min_dim"]), True)
                gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=ssd_dataset_collate)
                val_dataset = SSDDataset(self.lines[self.num_train:], (Config["min_dim"], Config["min_dim"]), False)
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=ssd_dataset_collate)

            for param in self.model.vgg.parameters():
                param.requires_grad = True

            for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
                self.fit_one_epoch(self.net, self.criterion, epoch, epoch_size, epoch_size_val, gen, gen_val,
                                   Unfreeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()
