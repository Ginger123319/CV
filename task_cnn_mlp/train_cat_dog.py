import torch
# from net import Net
from torch.utils.tensorboard import SummaryWriter

from data_cat_dog import CIFARDataset
from net_cat_dog import Net
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

DEVICE = "cuda"


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self, root):
        # 数据可视化工具使用
        self.writer = SummaryWriter("./log")

        self.train_data = CIFARDataset(root, True)
        # print(self.train_data.data.shape)
        # print(self.train_data.targets.shape)
        self.train_loader = DataLoader(self.train_data, batch_size=120, shuffle=True)

        self.test_data = CIFARDataset(root, False)
        self.test_loader = DataLoader(self.test_data, batch_size=60, shuffle=True)

        # 创建网络对象
        self.net = Net().to(DEVICE)
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())
        # 创建损失函数
        # loss = torch.mean((out - tag_data) ** 2)
        # self.loss_func = nn.MSELoss()  # 均方差损失函数
        # self.loss_func = nn.CrossEntropyLoss()  # 内部带有Softmax函数对输出进行归一化处理
        # self.loss_func = nn.NLLLoss()
        # self.loss_func = nn.BCEWithLogitsLoss()  # 内部带有Sigmoid函数对输出进行归一化处理
        self.loss_func = nn.BCELoss()

    def __call__(self, *args, **kwargs):
        # 训练
        for epoch in range(10000):
            # self.net.train()
            sum_loss = 0.
            for i, (images, tags) in enumerate(self.train_loader):
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
                out = self.net.forward(img_data).reshape(-1)
                # loss = torch.mean((out - tag_data) ** 2)
                loss = self.loss_func(out, tag_data)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                sum_loss = sum_loss + loss.item()

            train_avg_loss = sum_loss / len(self.train_loader)
            print(f"训练轮次：{epoch + 1}==========平均损失：{train_avg_loss}")

            test_sum_loss = 0.
            sum_score = 0.
            # 测试,输出与标签作比较，求精度
            for i, (images, tags) in enumerate(self.test_loader):
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
                test_out = self.net.forward(img_data).reshape(-1)
                # loss = torch.mean((out - tag_data) ** 2)
                test_loss = self.loss_func(test_out, tag_data)
                test_sum_loss = test_sum_loss + test_loss.item()
                # 把输出中数值最大处的索引取出来就是类别名称0~9
                # outs = torch.argmax(out, dim=1)
                outs = (test_out > 0.5).float()
                # print(outs.shape)
                # 转换后是长度为100的矢量
                # 从one-hot模式转换回来
                # tags = torch.argmax(tag_data, dim=1)
                # tag_data = one_hot(tags, 10)
                # 转换后是长度为100的矢量

                score = torch.mean(torch.eq(outs, tag_data).float())
                sum_score = sum_score + score.item()
            # print(sum_score)

            test_avg_loss = test_sum_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            # print(type(test_avg_score))
            print(f"测试轮次：{epoch + 1}==========平均损失：{test_avg_loss}")
            print(f"测试轮次：{epoch + 1}==========平均精度：{test_avg_score}")
            print()
            # # 使用writer收集标量数据
            # self.writer.add_scalars("loss", {"train_loss": train_avg_loss, "test_loss": test_avg_loss}, epoch + 1)
            # self.writer.add_scalar("score", test_avg_score, epoch + 1)


if __name__ == '__main__':
    train = Train(r"..\..\source\cat_dog\cat_dog")
    train()

# 错误1：AttributeError: 'Tensor' object has no attribute 'float32'
# score = torch.mean(out.eq(tag_data).float32)
# 需要将标签从one-hot模式转换回来
# 注意计算后的数据类型，与期望的数据类型是否一致，比如loss以及score
# 转为one-hot的函数需要两个参数，一个是输入，一个是矢量的元素个数
# 使用函数的时候，一个是确认函数是哪个库中的，另一个是确认函数需要的参数有哪些，需要看源码
