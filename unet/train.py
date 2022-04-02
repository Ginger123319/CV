import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from unet import UNet
from mydata import MyData
from torchvision.utils import save_image

path = r"..\..\source\VOCdevkit\VOC2012"
params_path = r"..\..\source\VOCdevkit\params_path"
img_save_path = r"..\..\source\VOCdevkit\savedImg"

net = UNet().cuda()
opt = torch.optim.Adam(net.parameters())
loss_func = nn.MSELoss()

data_loader = DataLoader(MyData(path), batch_size=1, shuffle=True)

# if os.path.exists(params_path):
#     net.load_state_dict(torch.load(params_path))
# else:
#     print("No params_path!")

# 如果文件夹不存在就创建一个目录
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

epoch = 1
while True:
    for i, (img, tag) in enumerate(data_loader):
        img = img.cuda()
        tag = tag.cuda()
        out = net(img)

        loss = loss_func(out, tag)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 5 == 0:
            print("epoch:{},count:{},loss:{}".format(epoch, i, loss.item()))
    # 取一张原图，一张输出图，一张标签图拼接到一起进行保存
    x = img[0]
    x_ = out[0]
    y = tag[0]
    img = torch.stack([x, x_, y], 0)
    save_image(img.cpu(), os.path.join(img_save_path, "{}.png".format(epoch)))
    epoch += 1
    torch.save(net.state_dict(), params_path)
    print("参数保存成功！")
