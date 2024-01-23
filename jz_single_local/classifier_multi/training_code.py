import os
import cv2
import copy
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from classifier_multi.early_stopping import EarlyStopping, calc_slope

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CSVDataset(Dataset):
    def __init__(self, input_df, class_name, input_shape, image_col, label_col, id_col, partition_dir, is_train=True):
        self.transform = []
        self.transform.append(transforms.ToTensor())
        if is_train:
            self.transform.append(transforms.RandomHorizontalFlip(0.5))
        self.transform = transforms.Compose(self.transform)

        self.data = []
        self.class_name = class_name
        self.input_shape = input_shape
        self.img_root = partition_dir
        df = input_df
        # 构建数据集
        for index, row in df.iterrows():
            if not pd.isna(row[label_col]):
                tag = self.class_name.index(row[label_col])
            else:
                # 如果没有标签，设置一个为0的假标签
                tag = 0
            self.data.append((row[id_col], row[image_col], tag))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_id = self.data[index][0]
        img_path = self.data[index][1]
        img_path = os.path.join(self.img_root, img_path)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        img = cv2.resize(img, self.input_shape)
        tag = self.data[index][2]
        return sample_id, self.transform(img), np.int64(tag), index


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self, train_dataset, val_dataset, model, batch_size, optimizer_type, epochs):

        self.epochs = epochs
        self.val = val_dataset

        # 打印训练集和验证集的比例
        if self.val is not None:
            print("train_size:{} val_size:{}".format(len(train_dataset), len(val_dataset)))

        # 加载训练和验证数据
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
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
        early_stopping = EarlyStopping(patience=7, verbose=True)

        best_w = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0

        point_list = []
        # 训练
        for epoch in range(self.epochs):

            sum_loss = 0.
            for i, (_, images, tags, _) in enumerate(self.train_loader):
                self.net.train()

                img_data = images.to(DEVICE)
                tag_data = tags.to(DEVICE)

                out = self.net.forward(img_data)
                loss = self.loss_func(out, tag_data)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()
                point_list.append(loss.item())

            train_avg_loss = sum_loss / len(self.train_loader)

            print(f"训练轮次：{epoch + 1}==========平均损失：{train_avg_loss}")
            # loss斜率早停
            if len(point_list) > 1:
                print("loss曲线斜率为：{}".format(abs(calc_slope(point_list))))
                if abs(calc_slope(point_list)) < 2e-2:
                    break

            if self.val is not None:
                test_sum_loss = 0.
                sum_score = 0.
                # 测试,输出与标签作比较，求精度
                for i, (_, images, tags, _) in enumerate(self.test_loader):
                    self.net.eval()

                    img_data = images.to(DEVICE)
                    tag_data = tags.to(DEVICE)

                    test_out = self.net.forward(img_data)
                    test_loss = self.loss_func(test_out, tag_data)

                    test_sum_loss = test_sum_loss + test_loss.item()
                    outs = torch.argmax(test_out, dim=1)

                    score = torch.mean(torch.eq(outs, tag_data).float())
                    sum_score = sum_score + score.item()

                test_avg_loss = test_sum_loss / len(self.test_loader)
                test_avg_score = sum_score / len(self.test_loader)

                early_stopping(test_avg_loss, self.net)
                # stop train
                if early_stopping.early_stop:
                    tqdm.write("Early stopping at {} epoch".format(epoch - 1))
                    break

                if test_avg_score >= best_acc:
                    best_acc = test_avg_score
                    best_w = copy.deepcopy(self.net.state_dict())

                print(f"测试轮次：{epoch + 1}==========平均损失：{test_avg_loss}")
                print(f"测试轮次：{epoch + 1}==========平均精度：{test_avg_score}")
                print()
        if self.val is not None:
            print("最佳精度为：{}".format(best_acc))
            return self.net, best_w
        return self.net, self.net.state_dict()
