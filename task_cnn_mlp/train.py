import torch
from net import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import one_hot

DEVICE = "cuda"


# 采用数据集的方式实现手写数字识别的训练和测试


class Train:
    def __init__(self):
        # 导入数据集，处理数据集中的data和targets
        # transforms.ToTensor()只是对数据进行基本的转换
        # 从PILImage转为tensor类型
        # 归一化处理，数据处理为0到1之间的浮点数
        # 换轴，从HWC->CHW
        # 不过只有在加载数据之后的调用过程中才会对图片进行实时处理
        # 数据也需要处理，需要转化（reshape）为全连接网络能够输入的形状
        # 标签的one-hot转换还没有做，需要保证输出的结果与标签的形状一致
        self.train_data = datasets.MNIST(root=r"..\..\source\MNIST_DATA", train=True, download=True,
                                         transform=transforms.ToTensor())
        # print(self.train_data.data.shape)
        # print(self.train_data.targets.shape)
        # # torch.Size([60000, 28, 28])
        # # torch.Size([60000])
        self.train_loader = DataLoader(self.train_data, batch_size=600, shuffle=True)

        self.test_data = datasets.MNIST(root=r"..\..\source\MNIST_DATA", train=False, download=False,
                                        transform=transforms.ToTensor())
        self.test_loader = DataLoader(self.test_data, batch_size=100, shuffle=True)

        # CIFAR10数据集
        self.train_data = datasets.CIFAR10(root=r"..\..\source\CIFAR10_DATA", train=True, download=True,
                                           transform=transforms.ToTensor())
        self.train_loader = DataLoader(self.train_data, batch_size=500, shuffle=True)

        self.test_data = datasets.CIFAR10(root=r"..\..\source\CIFAR10_DATA", train=False, download=False,
                                          transform=transforms.ToTensor())
        self.test_loader = DataLoader(self.test_data, batch_size=100, shuffle=True)
        print(self.test_data.data.shape)
        print(len(self.test_data.targets))

        # 创建网络对象
        self.net = Net().to(DEVICE)
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, *args, **kwargs):
        # 训练
        for epoch in range(10000):
            sum_loss = 0.
            sum_score = 0.
            for i, (images, tags) in enumerate(self.train_loader):
                # print(i)
                # 对图片数据进行处理
                # print(images.shape)
                # 全连接时使用
                # img_data = images.reshape(images.shape[0], -1)
                img_data = images
                img_data = img_data.to(DEVICE)

                # print(img_data.shape)
                # 对标签进行处理
                # print(tags.shape)
                # 转成one-hot模式

                tag_data = one_hot(tags, 10)
                tag_data = tag_data.to(DEVICE)

                # print(tags_data.shape)

                # 注意，输出时600*10，标签也是600*10，所以减法的结果时600*10
                # 结果无法求导，涉及损失函数时得求一个平均值
                out = self.net.forward(img_data)
                loss = torch.mean((out - tag_data) ** 2)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                sum_loss = sum_loss + loss.item()
            train_avg_loss = sum_loss / len(self.train_loader)
            print(f"训练轮次：{epoch}==========平均损失：{train_avg_loss}")

            # 测试,输出与标签作比较，求精度
            for i, (images, tags) in enumerate(self.test_loader):
                print(i)
                # 对图片数据进行处理
                # print(images.shape)

                # 全连接时使用
                # img_data = images.reshape(images.shape[0], -1)
                img_data = images
                img_data = img_data.to(DEVICE)

                # print(img_data.shape)
                # 对标签进行处理
                # print(tags.shape)
                # 转成one-hot模式

                tag_data = one_hot(tags, 10)
                tag_data = tag_data.to(DEVICE)

                # print(tags_data.shape)

                # 注意，输出时600*10，标签也是600*10，所以减法的结果时600*10
                # 结果无法求导，涉及损失函数时得求一个平均值
                out = self.net.forward(img_data)
                loss = torch.mean((out - tag_data) ** 2)
                sum_loss = sum_loss + loss.item()
                # 把输出中数值最大处的索引取出来就是类别名称0~9
                outs = torch.argmax(out, dim=1)
                # print(outs.shape)
                # 转换后是长度为100的矢量
                # 从one-hot模式转换回来
                tags = torch.argmax(tag_data, dim=1)
                # 转换后是长度为100的矢量

                score = torch.mean(torch.eq(outs, tags).float())
                sum_score = sum_score + score.item()
                # print(sum_score)

            test_avg_loss = sum_loss / len(self.test_loader)
            test_avg_score = sum_score / len(self.test_loader)
            # print(type(test_avg_score))
            print(f"测试轮次：{epoch}==========平均损失：{test_avg_loss}")
            print(f"测试轮次：{epoch}==========平均精度：{test_avg_score}")


if __name__ == '__main__':
    train = Train()
    train()

# 错误1：AttributeError: 'Tensor' object has no attribute 'float32'
# score = torch.mean(out.eq(tag_data).float32)
# 需要将标签从one-hot模式转换回来
# 注意计算后的数据类型，与期望的数据类型是否一致，比如loss以及score
# 转为one-hot的函数需要两个参数，一个是输入，一个是矢量的元素个数
# 使用函数的时候，一个是确认函数是哪个库中的，另一个是确认函数需要的参数有哪些，需要看源码
