import warnings
import numpy as np
import os 
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
        with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=3) as pbar:
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
    
                pbar.set_postfix(**{'loc_loss'  : loc_loss / (iteration + 1), 
                                    'conf_loss' : conf_loss / (iteration + 1),
                                    'lr'        : self.get_lr(self.optimizer)})
                pbar.update(1)
                 
        net.eval()
        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=3) as pbar:
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
    
                    self.optimizer.zero_grad()
                    
                    out = net(images)
                    loss_l, loss_c = criterion(out, targets)
                    loc_loss_val += float(loss_l.item())
                    conf_loss_val += float(loss_c.item())
    
                    pbar.set_postfix(**{'loc_loss'  : loc_loss_val / (iteration + 1), 
                                        'conf_loss' : conf_loss_val / (iteration + 1),
                                        'lr'        : self.get_lr(self.optimizer)})
                    pbar.update(1)
    
        total_loss = loc_loss + conf_loss
        val_loss = loc_loss_val + conf_loss_val
        
        writer.add_scalar('Train_loss', total_loss/(epoch_size+1), epoch)   
        writer.add_scalar('Val_loss', val_loss/(epoch_size_val+1), epoch)  
        
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ' - loss: %.3f  - val_loss: %.3f' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
        if epoch == Epoch - 1:
            save_path = self.work_dir+'/middle_dir/logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1), total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
            torch.save(self.model.state_dict(), save_path)
            from dc_model_repo.base.mr_log import logger
            logger.info("Model saved to: {}".format(save_path))


    def __init__(self, num_classes, image_size, num_train, num_val, lines, work_dir, tensorboard_dir, use_tfrecord, use_amp, mosaic=True):
        self.mosaic = mosaic
        Config["min_dim"] = image_size # 图片的形状
        Config["num_classes"] = num_classes + 1
        self.Cuda = torch.cuda.is_available() # 检测gpu是否可用
        self.use_tfrecord = use_tfrecord
        self.use_amp = use_amp
        if self.Cuda == False:
            self.use_amp = False
        print("Cuda:", self.Cuda, " use_amp:", self.use_amp, ' use_tfrecord:', self.use_tfrecord)
        
        self.work_dir = work_dir
        self.lines = lines
        self.num_val = num_val
        self.num_train = num_train
        self.model_path = self.work_dir + "/pre_training_weights/pre_training_weights.pth"
        
        self.model = get_ssd("train", Config["num_classes"])
    
        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.Cuda else 'cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished Loading weights!')

        if self.use_tfrecord:
            self.train_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord")
            self.train_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord.index")
            self.val_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord")
            self.val_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord.index")

        
        self.criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, self.Cuda)
        
        self.net = self.model.train()
        if self.Cuda:
            gpu_num = torch.cuda.device_count()
            device_ids = []
            for i in range(gpu_num):
                device_ids.append(i)
            self.net = torch.nn.DataParallel(self.model, device_ids=device_ids)
            cudnn.benchmark = True
            self.net = self.net.cuda()
        
        self.writer = SummaryWriter(log_dir=tensorboard_dir, flush_secs=60)
        if self.Cuda:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,Config["min_dim"],Config["min_dim"])).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,Config["min_dim"],Config["min_dim"])).type(torch.FloatTensor)
        self.writer.add_graph(self.model, (graph_inputs,))

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) 

    def get_data_loader(self, batch_size, schedule):
        if self.use_tfrecord:
            train_dataset = SSDDataset2(self.train_tf_record_file, self.train_tf_record_index, (Config["min_dim"], Config["min_dim"]), aug_config=schedule, mosaic=self.mosaic)
            gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)
            val_dataset = SSDDataset2(self.val_tf_record_file, self.val_tf_record_index, (Config["min_dim"], Config["min_dim"]))
            gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)
        else:
            train_dataset = SSDDataset(self.lines[:self.num_train], (Config["min_dim"], Config["min_dim"]), aug_config=schedule, mosaic=self.mosaic)
            gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)                
            val_dataset = SSDDataset(self.lines[self.num_train:], (Config["min_dim"], Config["min_dim"]))
            gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)
        return gen, gen_val

        
    def train(self, lr, freeze_epoch, total_epoch, optimizer, batch_size, schedule):
        
        epoch_size = max(1, self.num_train // batch_size)
        epoch_size_val = self.num_val // batch_size      
        
        if not isinstance(schedule, list):
            gen, gen_val = self.get_data_loader(batch_size=batch_size, schedule=schedule)
        else:
            schedule_list = list()
            for s in schedule:
                for _i in range(s.start, s.end):
                    schedule_list.append(s.config)

        if freeze_epoch > 0:
            lr = lr
            Init_Epoch = 0
            Freeze_Epoch = freeze_epoch
            
            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.92)
    
            for param in self.model.vgg.parameters():
                param.requires_grad = False
            
            for epoch in range(Init_Epoch, Freeze_Epoch):
                if isinstance(schedule, list):
                    gen, gen_val = self.get_data_loader(batch_size=batch_size, schedule=schedule_list[epoch])
                self.fit_one_epoch(self.net, self.criterion, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()
    
        if total_epoch - freeze_epoch > 0:
            lr = 1e-4
            Freeze_Epoch = freeze_epoch
            Unfreeze_Epoch = total_epoch
    
            if optimizer == "SGD":
                self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=0.92, verbose=True)
            
            for param in self.model.vgg.parameters():
                param.requires_grad = True
            for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
                if isinstance(schedule, list):
                        gen, gen_val = self.get_data_loader(batch_size=batch_size, schedule=schedule_list[epoch])
                self.fit_one_epoch(self.net, self.criterion, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, self.Cuda, self.writer)
                lr_scheduler.step()
        self.writer.close()

                
                