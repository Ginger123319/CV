import torch
from net import MnistNet
from data import MnistDataset
from torch.utils.data import DataLoader
from torch import optim


class Train:
    def __init__(self, root):
        # 训练集数据导入
        self.train_dataset = MnistDataset(root, True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=600, shuffle=True)

        # 测试集数据导入
        self.test_dataset = MnistDataset(root, False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 生成神经网络对象
        self.net = MnistNet()

        # 生成优化器
        # 此处无需传入步长，因为该优化器有一个默认的步长，并且会调整步长的大小
        # self.net.parameters()就是指的w，b之类的参数，说明该优化器是对这些参数进行优化的
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, *args, **kwargs):
        # 将训练集训练100000次
        # w = -dw * 0.1 + w
        # b = -db * 0.1 + b
        # 每次训练会将上一次训练好的参数代入到下一次训练的时候使用
        # 宏观的来看，每次训练完60000张图片后，会得到一批参数w和b
        # 到下一轮训练的时候，会将上一轮的参数代入到下一轮训练中
        # 因此多训练几轮的效果会比只训练一轮会好一些，但损失值也会在损失最低点附近震荡

        for epoch in range(100000):
            # 记录每一轮的损失值之和
            sum_loss = 0.
            # 每次取出600张图片和600个tag
            for i, (imgs, tags) in enumerate(self.train_loader):
                # 可以不写，默认会开启
                # 开始训练
                self.net.train()
                # 向网络中传入图片数据，得到输出，形状为（600，10）
                out = self.net.forward(imgs)
                # 设计损失函数
                # 由于涉及到求导的操作，需要对矩阵求一个平均值
                loss = torch.mean((out - tags) ** 2)

                # 开始学习
                # 清空梯度
                self.opt.zero_grad()
                # 对损失函数求当前的梯度
                loss.backward()
                # 对参数进行修改
                self.opt.step()

                # 将每个批次的损失都累加起来
                sum_loss = sum_loss + loss.item()
            avg_loss = sum_loss / len(self.train_loader)
            print("============>", avg_loss)

            # 开始测试测试集
            # 定义一个测试得分，也就是精确度
            sum_score = 0.
            # 可以反映出测试过程是否出现了过拟合现象
            # 过拟合时，曲线趋向于过每一个点，然而这种情况反映不出整体的趋势
            # 因此由这样训练出来的模型进行测试反而会让损失值升高
            test_sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.test_loader):
                # 开启测试模式
                self.net.eval()

                test_out = self.net.forward(imgs)
                test_loss = torch.mean((test_out - tags) ** 2)
                test_sum_loss = test_sum_loss + test_loss.item()

                # 精度计算
                # 将one-hot模式转回来
                # 将100，10的矩阵处理长度为100的矢量，存储的就是1轴上最大值的索引值，用来与测试集的标签做比较
                pre = torch.argmax(test_out, dim=1)
                # 将100，10的矩阵处理长度为100的矢量，存储的就是1轴上最大值【就是1】的索引值，用来和输出值比较
                label_tags = torch.argmax(tags, dim=1)
                # 一批数据的精度就等于输出正确的值除以总数
                score = torch.mean(torch.eq(pre, label_tags).float()).item()
                # 一轮测试的精度之和
                sum_score = sum_score + score
                # print("test_out:", pre[:10], "label_tags:", label_tags[:10])
            test_avg_loss = test_sum_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            print("epoch:", epoch, "test_loss:", test_avg_loss, "test_score:", test_avg_score)


if __name__ == '__main__':
    train = Train(r"..\..\source\MNIST_IMG")
    # 调用call函数
    train()
