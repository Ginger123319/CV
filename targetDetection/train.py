# from net import Net
from net_plus import Net
from dataDeal import MyData
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from utils import iou
import numpy as np
import torch
import os

# params_path = "../../source/target_detection/param"
params_path = "../../source/target_detection/param_plus"

# 导入训练集
train_data = MyData("../../source/target_detection", is_train=True)
train_loader = DataLoader(train_data, batch_size=60, shuffle=True)

# 网络相关实例化
net = Net().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()
loss_func2 = nn.BCELoss()

if os.path.exists(params_path):
    net.load_state_dict(torch.load(params_path))
    print("参数加载成功!")
else:
    print("No param!")

# 开始训练
for epoch in range(10):
    net.train()
    sum_loss = 0.
    sum_avg_iou = 0.
    out = 0.
    out2 = 0.
    label = 0.
    label2 = 0.
    real_len = len(train_loader)
    # for i, (img, label) in enumerate(train_loader):
    for i, (img, label, label2) in enumerate(train_loader):
        # print(f"第{i}批")
        # print(img.shape)
        # print(label.shape)
        img = img.cuda()
        label = label.cuda()
        label2 = label2.cuda()
        # out = net(img)
        out, out2 = net(img)

        loss = loss_func(out, label)
        # 保持两个值的形状一致
        out2 = out2.squeeze()
        loss2 = loss_func2(out2, label2)
        total_loss = loss2 + loss

        # 注意index为元组类型，需要取出第一个元素进行判断是否有值
        out2 = out2.detach().cpu()
        label2 = label2.detach().cpu()

        out = out.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        index = torch.where(out2 > 0.5)
        # print(index[0])
        if len(index[0]) == 0:
            avg_iou = 0
            real_len -= 1
        else:
            # print(index)
            out = out[index]
            label = label[index]
            avg_iou = np.mean(iou(out, label))

        opt.zero_grad()
        # loss.backward()
        total_loss.backward()
        opt.step()

        # sum_loss += loss.item()
        sum_loss += total_loss.item()
        sum_avg_iou += avg_iou.item()
    # print("out2 is {}".format(out2))
    # print("out is {}".format(out))
    # print("label is {}".format(label))

    # 计算分类精度
    out2 = (out2 > 0.5).float()
    accuracy = torch.mean((out2.eq(label2)).float())

    avg_loss = sum_loss / len(train_loader)
    print(f"第{epoch}轮次训练的平均损失为：{avg_loss}")
    print(f"第{epoch}轮次训练的平均IOU为：{sum_avg_iou / real_len}")
    print(f"第{epoch}轮次训练的最后一批的分类损失为：{loss2}")
    print(f"第{epoch}轮次训练的最后一批的分类精度为：{accuracy}")

    # 保存参数
    torch.save(net.state_dict(), params_path)
    print("参数保存成功！")
