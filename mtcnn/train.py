# 创建训练器----以训练三个网络


import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from sampling import FaceDataset  # 导入数据集


# 创建训练器
class Trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):  # 网络，参数保存路径，训练数据路径，cuda加速为True
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:  # 默认后面有个else
            self.net.cuda()  # 给网络加速

        # 创建损失函数
        # 置信度损失
        self.cls_loss_fn = nn.BCELoss()  # ★二分类交叉熵损失函数，是多分类交叉熵（CrossEntropyLoss）的一个特例；用BCELoss前面必须用sigmoid激活,用CrossEntropyLoss前面必须用softmax函数
        # 偏移量损失
        self.offset_loss_fn = nn.MSELoss()

        # 创建优化器
        self.optimizer = optim.Adam(self.net.parameters())

        # 恢复网络训练---加载模型参数，继续训练
        if os.path.exists(self.save_path):  # 如果文件存在，接着继续训练
            net.load_state_dict(torch.load(self.save_path))
            print("加载权重成功！")

    # 训练方法
    def train(self):
        faceDataset = FaceDataset(self.dataset_path)  # 数据集
        dataloader = DataLoader(faceDataset, batch_size=2000, shuffle=True, num_workers=0, drop_last=False)  # 数据加载器
        # num_workers=4：有4个线程在加载数据(加载数据需要时间，以防空置)；drop_last：为True时表示，防止批次不足报错。
        loss_flag = 1.
        # while True:
        for epoch in range(500):
            sum_loss = 0.
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                # 样本，置信度，偏移量
                if self.isCuda:  # cuda把数据读到显存里去了(先经过内存)；没有cuda在内存，有cuda在显存
                    img_data_ = img_data_.cuda()  # [512, 3, 12, 12]
                    category_ = category_.cuda()  # 512, 1]
                    offset_ = offset_.cuda()  # [512, 4]

                # 网络输出
                _output_category, _output_offset = self.net(img_data_)  # 输出置信度，偏移量
                # print(_output_category.shape)     # [512, 1, 1, 1]
                # print(_output_offset.shape)       # [512, 4, 1, 1]
                output_category = _output_category.reshape(-1, 1)  # [512,1]
                output_offset = _output_offset.reshape(-1, 4)  # [512,4]
                # output_landmark = _output_landmark.view(-1, 10)

                # 计算分类的损失----置信度
                category_mask = torch.lt(category_, 2)  # 对置信度小于2的正样本（1）和负样本（0）进行掩码; ★部分样本（2）不参与损失计算；符合条件的返回1，不符合条件的返回0
                category = torch.masked_select(category_, category_mask)  # 对“标签”中置信度小于2的选择掩码，返回符合条件的结果
                output_category = torch.masked_select(output_category, category_mask)  # 预测的“标签”进掩码，返回符合条件的结果
                cls_loss = self.cls_loss_fn(output_category, category)  # 对置信度做损失

                # 计算bound回归的损失----偏移量
                offset_mask = torch.gt(category_, 0)  # 对置信度大于0的标签，进行掩码；★负样本不参与计算,负样本没偏移量;[512,1]
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引；[244]
                offset = offset_[offset_index]  # 标签里饿偏移量；[244,4]
                output_offset = output_offset[offset_index]  # 输出的偏移量；[244,4]
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失

                # 总损失
                loss = cls_loss + offset_loss
                sum_loss += loss

                # 反向传播，优化网络
                self.optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()  # 计算梯度
                self.optimizer.step()  # 优化网络


                # 保存
                # if (i+1)/1000==0:
                #     torch.save(self.net.state_dict(), self.save_path) # state_dict保存网络参数，save_path参数保存路径
                #     print("save success")                            # 每轮次保存一次；最好做一判断：损失下降时保存一次
            avg_loss = sum_loss / len(dataloader)
            # 输出损失：loss-->gpu-->cup（变量）-->tensor-->array
            print("loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(),
                  " offset_loss",
                  offset_loss.cpu().data.numpy())
            print("第{0}轮的平均损失为{1}".format(epoch, avg_loss))
            if avg_loss < loss_flag:
                torch.save(self.net.state_dict(), self.save_path)  # state_dict保存网络参数，save_path参数保存路径
                print("save success")  # 每轮次保存一次；最好做一判断：损失下降时保存一次
                loss_flag = avg_loss
            else:
                loss_flag = loss_flag
            print(loss_flag)
# 备注：
# [1] num_workers:表示有多少线程在工作;
# [2] lt:小于；gt:大于；eq:等于；le：小于等于；ge:大于等于
# [3] loss.cpu().data.numpy():把损失从从“cuda”里放到“cpu”,在根据data属性转成numpy数据
