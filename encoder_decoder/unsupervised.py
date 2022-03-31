import torch
from torch import nn, optim
import thop
from thop import clever_format
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# 编解码网络模型

# 编码网络-提取特征
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10, bias=False),
            nn.BatchNorm1d(10)
        )

    def forward(self, x):
        return self.layer(x)


# 解码网络-还原特征
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(10, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784, bias=False),
            nn.BatchNorm1d(784)
        )

    def forward(self, x):
        return self.layer(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            Encoder(),
            Decoder()
        )
        # print(self.layer)

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    # 使用MNIST数据集进行测试
    # TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists;
    # found <class 'PIL.Image.Image'>
    # 取出来的数据类型默认是<class 'PIL.Image.Image'>，需要转为tensors
    # 使用torchvision中的transforms类
    train_data = datasets.MNIST(root=r"..\..\source\MNIST_DATA", train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    # 训练前预处理：定义网络对象，优化器对象，损失函数

    net = Net()

    opt = optim.Adam(net.parameters())
    # 返回一个函数，重命名为loss_fun
    loss_fun = nn.MSELoss()
    counter = 1
    for epoch in range(100):
        for i, (img_data, y) in enumerate(train_loader):
            # print(i)
            # 注意此处的形状为N CHW
            # 图片默认格式是N HWC，存在一个换轴操作或者reshape操作
            # print(img_data.shape)
            # print(y.shape)
            # 解码器使用全连接，就需要对取出的图片进行reshape操作
            img_data = img_data.reshape(img_data.shape[0], -1)
            # print(img_data.shape)
            # print(clever_format(thop.profile(net, (img_data,))))

            out = net.forward(img_data)
            # print(out.shape)

            loss = loss_fun(out, img_data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch > 0:
                # 每十个批次打印一次loss，保存一次输出结果
                if i % 10 == 0:
                    print(i)
                    print(loss.item())
                    fake_img = out.reshape(-1, 1, 28, 28)
                    real_img = img_data.reshape(-1, 1, 28, 28)
                    # 保存为10行，每行就有10张图片
                    save_image(fake_img, "../../img/fake_img{}.png".format(counter), nrow=10)
                    save_image(real_img, "../../img/real_img{}.png".format(counter), nrow=10)
                    counter += 1
