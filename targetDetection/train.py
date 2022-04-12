from net import Net
from dataDeal import MyData
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from utils import iou
import numpy as np
import torch
import os

params_path = "../../source/target_detection/param"

# 导入训练集
train_data = MyData("../../source/target_detection", is_train=True)
train_loader = DataLoader(train_data, batch_size=40, shuffle=True)


# 网络相关实例化
net = Net().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists(params_path):
    net.load_state_dict(torch.load(params_path))
    print("参数加载成功!")
else:
    print("No param!")

# 开始训练
for epoch in range(200):
    net.train()
    sum_loss = 0.
    out = 0.
    label = 0.
    for i, (img, label) in enumerate(train_loader):
        print(f"第{i}批")
        # print(img.shape)
        # print(label.shape)
        img = img.cuda()
        label = label.cuda()
        out = net(img)
        loss = loss_func(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

        sum_loss += loss.item()
    out = out.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    avg_iou = np.mean(iou(out, label))
    avg_loss = sum_loss / len(train_loader)
    print(f"第{epoch}轮次训练的平均损失为：{avg_loss}")
    print(f"第{epoch}轮次训练的平均IOU为：{avg_iou}")

    # 保存参数
    torch.save(net.state_dict(), params_path)
    print("参数保存成功！")
