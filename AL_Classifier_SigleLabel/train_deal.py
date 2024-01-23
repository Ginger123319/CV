import shutil
import torch
import os
# from net import Net
from torch.utils.tensorboard import SummaryWriter
import cfg
from data_deal import CSVDataset
from net_deal import _get_model
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from early_stopping import EarlyStopping
import numpy as np
from tqdm import tqdm
from sklearn import metrics
metrics.accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self, label_path, unlabeled_path, input_shape, model_type, num_class):
        # 数据可视化工具使用
        self.writer = SummaryWriter("./log")

        self.train_data = CSVDataset(label_path=label_path, unlabeled_path=unlabeled_path, input_shape=input_shape,
                                     is_train=True)
        # 将输入的训练集按8：2的方式划分为训练数据和验证数据
        train_size = int(0.8 * len(self.train_data))
        test_size = len(self.train_data) - train_size
        train_ds, val_ds = torch.utils.data.random_split(self.train_data, (train_size, test_size))

        # 加载训练和验证数据
        self.train_loader = DataLoader(train_ds, shuffle=True, batch_size=2, num_workers=0)
        self.test_loader = DataLoader(val_ds, shuffle=True, batch_size=16, num_workers=0)

        # print(len(train_ds), len(val_ds))
        # exit()

        # 创建网络对象
        self.net = _get_model(model_type=model_type, num_class=num_class).to(DEVICE)
        # 加载参数
        if os.path.exists(cfg.param_path):
            try:
                self.net.load_state_dict(torch.load(cfg.param_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(cfg.param_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())
        # self.opt = optim.SGD(self.net.parameters(),lr=0.01)
        # 创建损失函数
        # loss = torch.mean((out - tag_data) ** 2)
        # self.loss_func = nn.MSELoss()  # 均方差损失函数
        self.loss_func = nn.CrossEntropyLoss()  # 内部带有Softmax函数对输出进行归一化处理
        # self.loss_func = nn.NLLLoss()
        # self.loss_func = nn.BCEWithLogitsLoss()  # 内部带有Sigmoid函数对输出进行归一化处理
        # self.loss_func = nn.BCELoss()

    def __call__(self, *args, **kwargs):
        # 早停
        # to track the validation loss as the model trains
        valid_losses = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=5, verbose=True, save_path="./")
        # 训练
        for epoch in range(30):
            # self.net.train()
            sum_loss = 0.
            for i, (id, images, tags) in enumerate(self.train_loader):
                self.net.train()
                # print(i)
                # 对图片数据进行处理
                # print(images.shape)
                # 全连接时使用
                # img_data = images.reshape(images.shape[0], -1)
                img_data = images.to(DEVICE)

                # print(img_data.shape)
                # 对标签进行处理
                # print(tags.shape)
                # 转成one-hot模式

                # tag_data = one_hot(tags, 10)
                tag_data = tags.to(DEVICE)
                # tag_data = tag_data[:, None]

                # print(tags_data.shape)

                # 注意，输出时600*10，标签也是600*10，所以减法的结果时600*10
                # 结果无法求导，涉及损失函数时得求一个平均值
                out = self.net.forward(img_data)
                # loss = torch.mean((out - tag_data) ** 2)
                loss = self.loss_func(out, tag_data)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # sum_loss += loss.cpu().detach().item()
                sum_loss += loss.item()

            train_avg_loss = sum_loss / len(self.train_loader)
            # 保存参数
            torch.save(self.net.state_dict(), cfg.param_path)
            # torch.save(self.opt.state_dict(), cfg.opt_path)
            print(f"训练轮次：{epoch + 1}==========平均损失：{train_avg_loss}")

            test_sum_loss = 0.
            sum_score = 0.
            # 测试,输出与标签作比较，求精度
            for i, (id, images, tags) in enumerate(self.test_loader):
                self.net.eval()
                # print(i)
                # 对图片数据进行处理
                # print(images.shape)

                # 全连接时使用
                # img_data = images.reshape(images.shape[0], -1)
                img_data = images.to(DEVICE)

                # print(img_data.shape)
                # 对标签进行处理
                # print(tags.shape)
                # 转成one-hot模式

                # tag_data = one_hot(tags, 10)
                tag_data = tags.to(DEVICE)
                # tag_data = tag_data[:, None]
                # print(tags_data.shape)

                # 注意，输出时600*10，标签也是600*10，所以减法的结果时600*10
                # 结果无法求导，涉及损失函数时得求一个平均值
                test_out = self.net.forward(img_data)
                # loss = torch.mean((out - tag_data) ** 2)
                test_loss = self.loss_func(test_out, tag_data)
                valid_losses.append(test_loss.item())
                test_sum_loss = test_sum_loss + test_loss.item()
                # 把输出中数值最大处的索引取出来就是类别名称0~9
                outs = torch.argmax(test_out, dim=1)
                # outs = (test_out > 0.5).float()
                print(outs.shape)
                print(tag_data.shape)
                # 转换后是长度为100的矢量
                # 从one-hot模式转换回来
                # tags = torch.argmax(tag_data, dim=1)
                # tag_data = one_hot(tags, 10)
                # 转换后是长度为100的矢量

                score = torch.mean(torch.eq(outs, tag_data).float())
                sum_score = sum_score + score.item()
            # print(sum_score)

            # early_stop
            valid_loss = np.average(valid_losses)
            early_stopping(valid_loss, self.net)
            # stop train
            if early_stopping.early_stop:
                tqdm.write("Early stopping at {} epoch".format(epoch - 1))
                break

            test_avg_loss = test_sum_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            # print(type(test_avg_score))
            print(f"测试轮次：{epoch + 1}==========平均损失：{test_avg_loss}")
            print(f"测试轮次：{epoch + 1}==========平均精度：{test_avg_score}")
            print()
            # 使用writer收集标量数据
            self.writer.add_scalars("loss", {"train_loss": train_avg_loss, "test_loss": test_avg_loss}, epoch + 1)
            self.writer.add_scalar("score", test_avg_score, epoch + 1)


if __name__ == '__main__':
    input_label_path = 'temp.csv'
    input_unlabeled_path = 'empty.csv'
    # 删除log文件
    if os.path.exists(r"./log"):
        shutil.rmtree(r"./log")
        print("log is deleted！")

    train = Train(label_path=input_label_path, unlabeled_path=input_unlabeled_path, input_shape=(56, 56),
                  model_type="Resnet50", num_class=20)
    train()
