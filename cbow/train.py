import shutil
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import cfg
from cbow_dataset import CBowData
from net import CBowNet
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

DEVICE = "cuda"


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self, root):
        # 数据可视化工具使用
        self.writer = SummaryWriter("./log")

        self.train_data = CBowData(root)
        self.train_loader = DataLoader(self.train_data, batch_size=10, shuffle=True)

        # self.test_data = CIFARDataset(root, False)
        # self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=True)

        # 创建网络对象
        self.net = CBowNet().to(DEVICE)
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

        # self.opt = optim.SGD(self.net.parameters(),lr=0.01)
        # 加载参数
        if os.path.exists(cfg.param_path) and os.path.exists(cfg.opt_path):
            try:
                self.net.load_state_dict(torch.load(cfg.param_path))
                self.opt.load_state_dict(torch.load(cfg.opt_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(cfg.param_path)
                os.remove(cfg.opt_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")
        # 创建损失函数
        self.loss_func = nn.MSELoss()  # 均方差损失函数

    def __call__(self, *args, **kwargs):
        # 训练
        for epoch in range(100):
            # self.net.train()
            sum_loss = 0.
            for i, (images, tags) in enumerate(self.train_loader):
                self.net.train()
                img_data = images.to(DEVICE)
                tag_data = tags.to(DEVICE)

                out, tag = self.net.forward(img_data, tag_data)
                loss = self.loss_func(out, tag)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()

            train_avg_loss = sum_loss / len(self.train_loader)
            # 保存参数
            torch.save(self.net.state_dict(), cfg.param_path)
            torch.save(self.opt.state_dict(), cfg.opt_path)
            print(f"训练轮次：{epoch + 1}==========平均损失：{train_avg_loss}")
            print()
            # 使用writer收集标量数据
            self.writer.add_scalars("loss", {"train_loss": train_avg_loss}, epoch + 1)


if __name__ == '__main__':
    # 删除log文件
    if os.path.exists(r"./log"):
        shutil.rmtree(r"./log")
        print("log is deleted！")

    train = Train(cfg.train_dir)
    train()
