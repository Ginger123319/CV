import torch
from resnet50_p import Resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

DEVICE = "cuda"


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self):
        # CIFAR10数据集
        self.train_data = datasets.CIFAR10(root=r"..\..\source\CIFAR10_DATA", train=True, download=True,
                                           transform=transforms.ToTensor())
        self.train_loader = DataLoader(self.train_data, batch_size=500, shuffle=True)

        self.test_data = datasets.CIFAR10(root=r"..\..\source\CIFAR10_DATA", train=False, download=False,
                                          transform=transforms.ToTensor())
        self.test_loader = DataLoader(self.test_data, batch_size=100, shuffle=True)
        # print(self.test_data.data.shape)
        # print(len(self.test_data.targets))

        # 创建网络对象
        self.net = Resnet50().to(DEVICE)
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())
        # 创建损失函数
        self.loss_func = nn.CrossEntropyLoss()  # 内部带有Softmax函数对输出进行归一化处理

    def __call__(self, *args, **kwargs):
        # 训练
        for epoch in range(10000):
            # 记录训练的损失和
            sum_loss = 0.
            for i, (images, tags) in enumerate(self.train_loader):
                print(f"第 {i + 1} 批")
                self.net.train()
                img_data = images.to(DEVICE)
                tag_data = tags.to(DEVICE)

                out = self.net.forward(img_data)
                loss = self.loss_func(out, tag_data)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss = sum_loss + loss.item()

            train_avg_loss = sum_loss / len(self.train_loader)
            print(f"训练轮次：{epoch}==========平均损失：{train_avg_loss}")

            # 测试,输出与标签作比较，求精度
            test_sum_loss = 0.
            sum_score = 0.
            for i, (images, tags) in enumerate(self.test_loader):
                self.net.eval()

                img_data = images.to(DEVICE)

                tag_data = tags.to(DEVICE)
                print(tag_data.shape)

                test_out = self.net.forward(img_data)

                test_loss = self.loss_func(test_out, tag_data)
                test_sum_loss = test_sum_loss + test_loss.item()
                # 把输出中数值最大处的索引取出来就是类别名称0~9
                outs = torch.argmax(test_out, dim=1)
                print(outs.shape)
                # 转换后是长度为100的矢量

                score = torch.mean(torch.eq(outs, tag_data).float())
                sum_score = sum_score + score.item()

            test_avg_loss = test_sum_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            # print(type(test_avg_score))
            print(f"测试轮次：{epoch}==========平均损失：{test_avg_loss}")
            print(f"测试轮次：{epoch}==========平均精度：{test_avg_score}")
            print()


if __name__ == '__main__':
    train = Train()
    train()
