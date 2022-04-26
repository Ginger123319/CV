from torch import nn
from torch.utils.tensorboard import SummaryWriter

import dataset
from module import *
import torch
import os

param_path = r"D:\Python\source\YOLO\param.pt"


def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)  # N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,13,13,3,15
    # print("output:",output.shape)
    mask_obj = target[..., 0] > 0  # N,13,13,3
    # print("mask_obj:",mask_obj.shape)
    # mask_noobj = target[..., 0] == 0
    # print("mask_noobj:",mask_noobj.shape)
    # print("output[mask_obj]:",output[mask_obj].shape)
    # print("output[mask_noobj]:", output[mask_noobj].shape)
    # 置信度损失：需要的是正负样本的置信度。
    # loss_obj = torch.mean((output[mask_obj] - target[mask_obj]) ** 2)#N,15
    # loss_noobj = torch.mean((output[mask_noobj] - target[mask_noobj]) ** 2)
    # loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    # return loss
    c_loss_func = nn.BCEWithLogitsLoss()
    off_loss_func = nn.MSELoss()
    cls_loss_func = nn.CrossEntropyLoss()
    # 置信度损失：
    c_loss = c_loss_func(output[..., 0], target[..., 0])
    # 偏移量损失
    # print(output.shape)
    # print(output[mask_obj].shape)
    # print(output[mask_obj][:,1:5])
    # 交叉熵的两个参数分别为浮点数和长整型的long
    off_loss = off_loss_func(output[mask_obj][:, 1:5].float(), target[mask_obj][:, 1:5].float())

    # 多分类损失
    # print(target[mask_obj][:,5:])
    # print(torch.argmax(target[mask_obj][:,5:],dim=1))
    # cls_loss = cls_loss_func(output[mask_obj][:, 5:], torch.argmax(target[mask_obj][:, 5:], dim=1))
    # 分类标签没有使用one-hot
    # 需要把标签也转为Tensor float32的数据类型
    cls_loss = cls_loss_func(output[mask_obj][:, 5:], target[mask_obj][:, 5].long())
    loss = alpha * c_loss + (1 - alpha) * (off_loss + cls_loss)
    return loss


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = Darknet53().cuda()
    # 判断是否有权重，有就加载权重
    # 不适用于给一个异常的预权重且名字为param_path+ "_old"的情况
    # old和new有四种组成：空，old，new，old和new
    # 在保存结果异常的时候会启用old，如果old不存在则重新训练，所以不会出现将异常的结果重命名的情况
    # 只有old时也启用old（此处old一定是一个正常保存的参数）
    if os.path.exists(param_path):
        # torch.load()
        try:
            net.load_state_dict(torch.load(param_path))
            print("New param is loaded")
            if os.path.exists(param_path + "_old"):
                os.remove(param_path + "_old")
        except RuntimeError as r:
            print("New param load error")
            os.remove(param_path)
            if os.path.exists(param_path + "_old"):
                os.rename(param_path + "_old", param_path)
                net.load_state_dict(torch.load(param_path))
                print("Old param is loaded")
            else:
                print("没有可以回退的参数，将开始重新训练")
                print("error is", r)
    elif os.path.exists(param_path + "_old"):
        net.load_state_dict(torch.load(param_path + "_old"))
        os.rename(param_path + "_old", param_path)
        print("Only old param is loaded")
    else:
        print("NO Param!")

    net.train()
    opt = torch.optim.Adam(net.parameters())
    count = 1
    while count <= 10000:
        sum_loss = 0.
        for target_13, target_26, target_52, img_data in train_loader:
            output_13, output_26, output_52 = net(img_data.cuda())
            # print(output_13.shape, output_13.dtype)
            loss_13 = loss_fn(output_13, target_13.cuda(), 0.9)
            loss_26 = loss_fn(output_26, target_26.cuda(), 0.9)
            loss_52 = loss_fn(output_52, target_52.cuda(), 0.9)

            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print(loss.item())
            sum_loss += loss
        # 每训练一轮，保存一次权重,删除上一轮保留的权重
        avg_loss = sum_loss / len(train_loader)
        print("epoch {} avg loss is: {}".format(count, avg_loss))
        if os.path.exists(param_path):
            os.rename(param_path, param_path + "_old")
        # print(param_path)
        torch.save(net.state_dict(), param_path)
        # print("save success!")
        if os.path.exists(param_path + "_old"):
            os.remove(param_path + "_old")
        # print("delete old param success")
        count += 1
