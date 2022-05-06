import os
import shutil
import torch
from torch import optim
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter

# from dataset import MyData
from dataset_changed import MyData
# from net import Net
from net2 import Net
from torch.utils.data import DataLoader

param_path = r"param_pt"
log_path = r"./log"


class Trainer:
    def __init__(self):
        # 图形化展示损失
        self.writer = SummaryWriter(log_path)
        # 加载数据集
        # 增样的训练集效果更好，用真实的训练集训练反而精度在下降
        self.train_loader = DataLoader(MyData(True, False), batch_size=40, shuffle=True)
        self.test_loader = DataLoader(MyData(False), batch_size=50, shuffle=True)
        # 加载网络
        self.net = Net()
        self.opt = optim.SGD(self.net.parameters(), lr=0.01)
        # self.opt = optim.Adam(self.net.parameters())
        self.loss_fun = BCELoss()
        # 加载参数
        if os.path.exists(param_path):
            try:
                self.net.load_state_dict(torch.load(param_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(param_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")

    def train(self):
        # 开始训练
        save_flag = 0.
        for epoch in range(1000):
            self.net.train()
            sum_loss = 0.
            sum_test_loss = 0.
            sum_score = 0.
            for i, (data, tag, _) in enumerate(self.train_loader):
                # print(data.shape)
                # print(tag)
                # exit()
                # 新加的
                data = data.permute(0, 3, 1, 2)
                out = self.net(data)[0]
                # print(out.squeeze().shape)
                # exit()
                loss = self.loss_fun(out, tag.unsqueeze(dim=1).float())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()
                # print(loss.item())
                # exit()
            self.net.eval()
            for i, (test_data, test_tag, _) in enumerate(self.test_loader):
                # 新加的
                test_data = test_data.permute(0, 3, 1, 2)
                out = self.net(test_data)[0]
                loss = self.loss_fun(out.squeeze(), test_tag.float())
                sum_test_loss += loss.item()
                # 精度计算
                score = torch.mean((torch.eq((out.squeeze() > 0.9).float(), test_tag.float())).float())
                sum_score += score.item()
                # print(torch.mean((torch.eq((out.squeeze() > 0.5).float(), test_tag.float())).float()))
                # print(test_tag.float())
                # exit()
            # 训练
            avg_loss = sum_loss / len(self.train_loader)
            print("epoch {} avg_loss is {}".format(epoch, avg_loss))
            # 测试
            test_avg_loss = sum_test_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            print("epoch {} test_avg_loss is {}".format(epoch, test_avg_loss))
            print("epoch {} test_avg_score is {}".format(epoch, test_avg_score))
            self.writer.add_scalars("loss", {"avg_loss": avg_loss, "test_avg_loss": test_avg_loss}, epoch)
            self.writer.add_scalar("test_avg_score", test_avg_score, epoch)
            # 保存参数
            if save_flag <= test_avg_score:
                torch.save(self.net.state_dict(), param_path)
                save_flag = test_avg_score
                print("save success!")


if __name__ == '__main__':
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    train = Trainer()
    train.train()
