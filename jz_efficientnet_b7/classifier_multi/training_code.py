import os
import copy
import torch
import shutil
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from classifier_multi.early_stopping import EarlyStopping, calc_slope

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def divide_or_round_up(x, y):
    """
    判断一个数是否能够整除另一个数，如果能，返回结果；
    如果不能，返回整除后的整数部分并且加一。
    """
    if x % y == 0:
        # 如果能够整除，返回结果
        return x // y
    else:
        # 如果不能整除，返回整除后的整数部分加一
        return x // y + 1


# 自定义Dataset
class CSVDataset(Dataset):
    def __init__(self, input_df, class_name, input_shape, image_col, label_col, id_col, is_train=True):
        train_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.AutoAugment(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor()
        ])
        if is_train:
            self.transform = train_transform
        else:
            self.transform = test_transform

        self.data = []
        self.class_name = class_name
        df = input_df
        # 构建数据集
        for index, row in df.iterrows():
            if not pd.isna(row[label_col]):
                tag = self.class_name.index(row[label_col])
            else:
                # 如果没有标签，设置一个为-1的假标签
                tag = -1
            self.data.append([row[id_col], row[image_col], tag])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_id = self.data[index][0]
        img_path = self.data[index][1]

        img = Image.open(img_path).convert('RGB')
        tag = self.data[index][2]
        return sample_id, self.transform(img), np.int64(tag), index


# 自定义训练
class Train:
    def __init__(self, train_dataset, val_dataset, model, batch_size, optimizer_type, epochs):

        self.epochs = epochs
        self.val = val_dataset
        self.batch_size = batch_size
        if os.path.exists('logs'):
            shutil.rmtree('logs')
        self.writer = SummaryWriter(log_dir='logs', flush_secs=60)

        # 打印训练集和验证集的比例
        if self.val is not None:
            print("train_size:{} val_size:{}".format(len(train_dataset), len(val_dataset)))
        self.train_epoch_size = divide_or_round_up(len(train_dataset), batch_size)
        self.val_epoch_size = divide_or_round_up(len(val_dataset), batch_size)

        # 加载训练和验证数据
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
        if val_dataset is not None:
            self.test_loader = DataLoader(val_dataset, shuffle=False, batch_size=16, num_workers=0)

        # 创建网络对象
        self.net = model.to(DEVICE)

        # 创建优化器
        if optimizer_type == "Adam":
            self.opt = optim.Adam(self.net.parameters(), weight_decay=5e-4)
        elif optimizer_type == "SGD":
            self.opt = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=5e-4)
        else:
            raise Exception("不支持这种类型的优化器：{}".format(optimizer_type))

        # 创建损失函数
        self.loss_func = nn.CrossEntropyLoss()  # 内部带有Softmax函数对输出进行归一化处理

    def __call__(self, *args, **kwargs):

        # 早停
        # initialize the early_stopping object
        weights_path = args[0]
        early_stopping = EarlyStopping(save_path=weights_path, patience=7, verbose=True)

        best_w = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0

        point_list = []
        # 训练
        for epoch in range(self.epochs):

            sum_loss = 0.
            with tqdm(total=self.train_epoch_size, desc=f'Epoch {epoch + 1}/{self.epochs}', postfix=dict,
                      mininterval=0.3) as pbar:
                for iteration, (_, images, tags, _) in enumerate(self.train_loader):
                    self.net.train()
                    # 计算有标签损失
                    img_data = images.to(DEVICE)
                    tag_data = tags.to(DEVICE)

                    out = self.net.forward(img_data)
                    loss = self.loss_func(out, tag_data)

                    # 计算总损失
                    total_loss = loss

                    self.opt.zero_grad()
                    total_loss.backward()
                    self.opt.step()

                    sum_loss += total_loss.item()
                    point_list.append(total_loss.item())

                    # 设置tqdm打印信息
                    pbar.set_postfix(**{'train_avg_loss': sum_loss / (iteration + 1)})
                    pbar.update(1)

            train_avg_loss = sum_loss / (iteration + 1)
            self.writer.add_scalar('Train_loss', train_avg_loss, epoch)

            if self.val is None:
                # 训练loss斜率早停
                if len(point_list) > 1:
                    print("loss曲线斜率为：{}".format(abs(calc_slope(point_list))))
                    if abs(calc_slope(point_list)) < 2e-2:
                        break

            if self.val is not None:
                test_sum_loss = 0.
                sum_score = 0.
                # 测试,输出与标签作比较，求精度

                with tqdm(total=self.val_epoch_size, desc=f'Epoch {epoch + 1}/{self.epochs}', postfix=dict,
                          mininterval=0.3) as pbar:
                    for iteration, (_, images, tags, _) in enumerate(self.test_loader):
                        self.net.eval()

                        img_data = images.to(DEVICE)
                        tag_data = tags.to(DEVICE)

                        test_out = self.net.forward(img_data)
                        test_loss = self.loss_func(test_out, tag_data)

                        test_sum_loss = test_sum_loss + test_loss.item()
                        outs = torch.argmax(test_out, dim=1)

                        score = torch.mean(torch.eq(outs, tag_data).float())
                        sum_score += score.item()
                        # 设置tqdm打印信息
                        pbar.set_postfix(**{'test_avg_score': sum_score / (iteration + 1),
                                            'val_avg_loss': test_sum_loss / (iteration + 1)})
                        pbar.update(1)

                    test_avg_loss = test_sum_loss / (iteration + 1)
                    test_avg_score = sum_score / (iteration + 1)
                    self.writer.add_scalars('Val', {'val_loss': test_avg_loss, 'val_acc': test_avg_score},
                                            epoch)

                    early_stopping(test_avg_loss, self.net)
                    # early_stop train
                    if early_stopping.early_stop:
                        tqdm.write("Early stopping at {} epoch".format(epoch - 1))
                        break

                    if test_avg_score >= best_acc:
                        best_acc = test_avg_score
                        best_w = copy.deepcopy(self.net.state_dict())
        if self.val is not None:
            print("最佳精度为：{}".format(best_acc))
            self.net.load_state_dict(best_w)
        return self.net
