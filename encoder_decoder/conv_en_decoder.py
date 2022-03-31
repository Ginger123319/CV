# import torch
# from torch.nn.functional import interpolate
#
# # test = torch.randn(1, 1, 3, 3)
# # result = interpolate(test, scale_factor=2, mode='nearest')
# # print(result.shape)


from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F

LEARNING_RATE = 1e-3


# 编解码网络模型

# 编码网络-提取特征
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3
        )
        self.enc2 = nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=3
        )
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=4, out_channels=8, kernel_size=3
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=8, out_channels=1, kernel_size=3
        )

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x


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

    net = Autoencoder()
    # print(net)

    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE)
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
            # img_data = img_data.reshape(img_data.shape[0], -1)
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
                    # print(i)
                    print(loss.item())
                    fake_img = out.view(out.size(0), 1, 28, 28)
                    real_img = img_data.reshape(-1, 1, 28, 28)
                    # 保存为10行，每行就有10张图片
                    save_image(fake_img, "../../img/fake_img{}.png".format(counter), nrow=10)
                    save_image(real_img, "../../img/real_img{}.png".format(counter), nrow=10)
                    counter += 1
