from model_nets.training import Train
import os 
import re 
import shutil

def train(num_classes, image_size, val_size, Cosine_lr, mosaic, smooth_label, lr, freeze_epoch, total_epoch, optimizer, batch_size, work_dir, tensorboard_dir):
    training = Train(num_classes, image_size, val_size, Cosine_lr, mosaic, smooth_label, work_dir, tensorboard_dir )
    training.train(lr, freeze_epoch, total_epoch, optimizer, batch_size)
        
    # 筛选出val loss最低的一个pth文件
    logs_list_dir = os.listdir(work_dir+'/middle_dir/logs')
    val_dict = {} # 组成{pth文件名：val loss值}形式的字典
    val_list = [] # 组成[val loss值]形式的列表
    for i in logs_list_dir:
        if i.endswith('.pth'):
            val_loss = float(re.match(r'Epoch(.+)-Total_Loss(.+)-Val_Loss(.+).pth', i).group(3))
            val_dict[i] = val_loss
            val_list.append(val_loss)
    
    val_list.sort()
    
    val_list_1 = val_list[:1] # 筛选出val loss从小到大排列的第一名
    val_key_list_1 = [] # 将val loss排第一名的pth文件名保存在列表里
    for i in val_dict.keys():
        if val_dict[i] in val_list_1:
            val_key_list_1.append(i)
    for i in val_key_list_1:
        shutil.copyfile(work_dir+'/middle_dir/logs/'+i, work_dir+'/middle_dir/normal_train_best_model_dir/'+i)
    return val_key_list_1[0]