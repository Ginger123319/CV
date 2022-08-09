import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MyDatasets1 import MyDataset
from Net_Module import MainNet


# ' torch.argmax(output, dim=1)'
class Trainner:  # 定义训练类
    def __init__(self):
        self.save_path = 'models/pet1.pth'  # 实例化保存的地址
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断用cuda
        self.net = MainNet().to(self.device)  # 实例化网络
        if os.path.exists(self.save_path):  # 判断是否有之前训练过的网络参数，有的话就加载参数接着训练
            self.net.load_state_dict(torch.load(self.save_path))
            print("param loading!")
        self.traindata = MyDataset()  # 实例化制作的数据集
        self.trainloader = DataLoader(self.traindata, batch_size=4, shuffle=True)  # 加载数据集
        self.conf_loss_fn = nn.BCEWithLogitsLoss()  # 定义自信度的损失函数，这里可以用bceloss，不过bceloss要用sinmoid函数激活，用这个bce损失函数不需要用其激活
        self.crood_loss_fn = nn.MSELoss()  # 定义偏移量的均方差损失函数
        self.cls_loss_fn = nn.CrossEntropyLoss()  # 定义多分类的交叉熵损失函数
        self.optimzer = optim.Adam(self.net.parameters())  # 定义网络优化器

    def loss_fn(self, output, target, alpha):  # 定义损失函数，并传入三个参数，网络输出的数据，标签，和用来平衡正负样本损失侧重那个的权重
        # 数据形状[N,C,H,W]-->>标签形状[N,H,W,C]
        output = output.permute(0, 2, 3, 1)  # 将形状转成和标签一样
        # 通过reshape变换形状 [N,H,W,C]即[N,H,W,45]-->>[N,H,W,3,15]，分成3个建议框每个建议框有15个值
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output = output.cuda().double()  # 将数据转入cuda并转成double()，因为交叉熵输入类型是double双精度类型
        mask_obj = target[..., 0] > 0  # 拿到自信大于0的掩码值返回其索引，省略号代表前面不变在最后的15个值里面取第0个，第0个敢为自信度
        output_obj = output[mask_obj]  # 通过掩码获取对应位置的输出值
        target_obj = target[mask_obj]  # 通过掩码获取对应位置的标签
        loss_obj_conf = self.conf_loss_fn(output_obj[:, 0], target_obj[:, 0])  # 算自信都损失
        loss_obj_crood = self.crood_loss_fn(output_obj[:, 1:5], target_obj[:, 1:5])  # 算偏移量损失
        loss_obj_cls = self.cls_loss_fn(output_obj[:, 5:], torch.argmax(target_obj[:, 5:], dim=1))  # 可以用这个也可以用下面这个,
        # 或用下面这个也可以，用下面这个不需要对标签取最大值所以，用上面这个要对标签取最大值所以，因为输出是标量
        # loss_obj_cls=self.conf_loss_fn(output_obj[:,5:],target_obj[:,5:])
        loss_obj = loss_obj_conf + loss_obj_crood + loss_obj_cls  # 正样本总的损失

        mask_noobj = target[..., 0] == 0  # 获取负样本的掩码
        output_noobj = output[mask_noobj]  # 根据掩码获取数据
        target_noobj = target[mask_noobj]  # 根据掩码获取标签
        loss_noobj = self.conf_loss_fn(output_noobj[:, 0], target_noobj[:, 0])  # 计算负样本损失，负样本自信度为0，因此这里差不多就是和0 做损失
        loss = alpha * loss_obj + (1 - alpha) * loss_noobj  # 这里权重调整正负样本训练程度，如果负样本多久可以将权重给大点，负样本少就可以把权重给小点
        print("loss_noobj:", loss_noobj)
        return loss  # 将损失返回给调用方

    def train(self):  # 定义训练函数
        self.net.train()  # 这个可用可不用表示训练，网络如果用到batchnormal及dropout等，但测试里面必须添加self.net.eveal()
        epochs = 0  # 训练批次
        while True:
            print(epochs)
            for target_13, target_26, target_52, img_data in self.trainloader:  # 循环trainloader，将3个返回值分别复制给对应变量
                target_13, target_26, target_52, img_data = target_13.to(self.device), target_26.to(
                    self.device), target_52.to(self.device), img_data.to(self.device)  # 将数据和标签转入cuda
                output_13, output_26, output_52 = self.net(img_data)  # 将数据传入网络获得输出
                loss_13 = self.loss_fn(output_13, target_13, 0.7)  # 自信度损失
                loss_26 = self.loss_fn(output_26, target_26, 0.7)  # 偏移量损失
                loss_52 = self.loss_fn(output_52, target_52, 0.7)  # 分类损失

                loss = loss_13 + loss_26 + loss_52  # 总损失
                self.optimzer.zero_grad()  # 清空梯度
                loss.backward()  # 反向求导
                self.optimzer.step()  # 更新梯度
                print(loss.item())
                # torch.save(self.net.state_dict(),self.save_path)
            if epochs % 10 == 0:
                torch.save(self.net.state_dict(), self.save_path.format(epochs))  # 保存网络参数
            epochs += 1


if __name__ == '__main__':
    obj = Trainner()
    obj.train()
