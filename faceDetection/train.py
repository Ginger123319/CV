from PIL import ImageDraw
from PIL import Image

from net import Net
from dataDeal import MyData
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from utils import iou
import numpy as np
import torch
import os

params_path = "../../source/FACE/param"

# 导入训练集
train_data = MyData(r"D:\Python\source\FACE\celebA", is_train=True)
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
    sum_loss = 0.
    out = 0.
    label = 0.
    for i, (img, label) in enumerate(train_loader):
        net.train()
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

        # 开始画框【此处操作不熟悉，比如多维张量的索引取值label[0]】
        out = out.detach().cpu().numpy() * 300
        label = label.detach().cpu().numpy() * 300
        img = img.permute(0, 2, 3, 1)
        img = img.detach().cpu().numpy()
        img_data = np.array(img[0] * 255, dtype=np.uint8)
        img = Image.fromarray(img_data, "RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(np.array(label[0]), outline="red", width=2)
        draw.rectangle(np.array(out[0]), outline="yellow", width=2)
        # img.show()
        if i % 10 == 0:
            img.save(r"D:\Python\source\FACE\train_result\{0}.{1}.png".format(epoch, i + 1))
    print("该轮图片结果全部已经保存！")
    # out = out.detach().cpu().numpy()
    # label = label.detach().cpu().numpy()
    avg_iou = np.mean(iou(out, label))
    avg_loss = sum_loss / len(train_loader)
    print(f"第{epoch}轮次训练的平均损失为：{avg_loss}")
    print(f"第{epoch}轮次训练的平均IOU为：{avg_iou}")

    # 保存参数
    torch.save(net.state_dict(), params_path)
    print("参数保存成功！")
