import torch

from net_p import MnistNet
from torch.utils.data import DataLoader
from data_p import MnistDataset
from torch import optim


# 创建训练类
class Train:
    def __init__(self, root):
        # 导入并加载训练和测试集
        self.train_data = MnistDataset(root)
        # 数据集都需要打乱，参数shuffle为True
        self.train_loader = DataLoader(self.train_data, batch_size=600, shuffle=True)
        self.test_data = MnistDataset(root, False)
        # 数据集都需要打乱，参数shuffle为True
        self.test_loader = DataLoader(self.test_data, batch_size=100, shuffle=True)
        # 创建网络对象
        self.net = MnistNet()
        # 创建优化器，传入需要调整的参数
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, *args, **kwargs):
        # 开始训练
        for epoch in range(100000):
            # 用来累加每一批次的损失值
            sum_loss = 0.
            for i, (img_data, tags) in enumerate(self.train_loader):
                print(i)
                # 将图像数据输入到网络中得到输出
                out = self.net.forward(img_data)
                # 设计损失函数
                loss = torch.mean((out - tags) ** 2)
                # 清空梯度
                self.opt.zero_grad()
                # 求梯度
                loss.backward()
                # 减小梯度-也就是降低损失
                self.opt.step()
                sum_loss = sum_loss + loss
            # len(self.train_loader)就是内层循环一次的次数
            avg_loss = sum_loss / len(self.train_loader)
            print(f"epoch: {epoch} avg_loss: {avg_loss}")


if __name__ == '__main__':
    train = Train(r"..\source\MNIST_IMG")
    train()

    # 没有对输入数据做归一化处理
    # 没有对每一轮的形状进行判定
    # 在书写代码时需要边写便验证，不能写完了才验证
    # 错误1--RuntimeError: grad can be implicitly created only for scalar outputs
    # 只能对标量求导，然而直接计算的loss是一个矩阵，所以要求一个平均值
    # 错误2--UserWarning: Implicit dimension choice for softmax has been deprecated.
    # Change the call to include dim=X as an argument.
    # Softmax()没有给参数
