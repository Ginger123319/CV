from torch import nn

import dataset
from module import *
import torch


def loss_fn(output, target, alpha):

    output = output.permute(0, 2, 3, 1)#N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#N,13,13,3,15
    # print("output:",output.shape)
    mask_obj = target[..., 0] > 0#N,13,13,3
    # print("mask_obj:",mask_obj.shape)
    # mask_noobj = target[..., 0] == 0
    # print("mask_noobj:",mask_noobj.shape)
    # print("output[mask_obj]:",output[mask_obj].shape)
    # print("output[mask_noobj]:", output[mask_noobj].shape)
    #置信度损失：需要的是正负样本的置信度。
    # loss_obj = torch.mean((output[mask_obj] - target[mask_obj]) ** 2)#N,15
    # loss_noobj = torch.mean((output[mask_noobj] - target[mask_noobj]) ** 2)
    # loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    # return loss
    c_loss_func = nn.BCEWithLogitsLoss()
    off_loss_func = nn.MSELoss()
    cls_loss_func = nn.CrossEntropyLoss()
    #置信度损失：
    c_loss = c_loss_func(output[...,0],target[...,0])
    #偏移量损失
    # print(output.shape)
    # print(output[mask_obj].shape)
    # print(output[mask_obj][:,1:5])
    off_loss = off_loss_func(output[mask_obj][:,1:5].float(),target[mask_obj][:,1:5].float())

    #多分类损失
    # print(target[mask_obj][:,5:])
    # print(torch.argmax(target[mask_obj][:,5:],dim=1))
    cls_loss = cls_loss_func(output[mask_obj][:,5:],torch.argmax(target[mask_obj][:,5:],dim=1))
    loss = alpha * c_loss+(1-alpha)*(off_loss+cls_loss)
    return loss


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = Darknet53().cuda()
    net.train()

    opt = torch.optim.Adam(net.parameters())

    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            output_13, output_26, output_52 = net(img_data.cuda())
            loss_13 = loss_fn(output_13, target_13.cuda(), 0.9)
            loss_26 = loss_fn(output_26, target_26.cuda(), 0.9)
            loss_52 = loss_fn(output_52, target_52.cuda(), 0.9)

            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss.item())
