import shutil
import sys
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cfg
import dataset
# from module import *
from yolobody import YoloBody
import torch
import os
import numpy
from acc_cal import _target_cal

<<<<<<< HEAD
param_path = r"D:\Python\source\key_mouse\param.pt"
=======
param_path = r"..\..\source\key_mouse\param.pt"
>>>>>>> 8e573c5ca72d22baed81e334b509533fd6d7a85a


def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)  # N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,13,13,3,15
    # print("output:",output.shape)
    # 辅助选出正样本，mask_obj作为索引
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
    # 置信度实际有两层含义：一是判断一个格子上是否有目标，训练上的二分类问题；二是测试的时候用它的大小来选择合适的建议框
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
    # alpha为权重，权重大的就会加强训练，因为正负样本不均衡，所以需要用权重来处理不好训练的分类损失
    # 偏移量训练和分类的训练相对来说比较简单，因为只会取正样本来做损失，受到训练样本不均衡的问题影响相对小
    off_loss = off_loss_func(output[mask_obj][:, 1:5].float(), target[mask_obj][:, 1:5].float())

    # 多分类损失
    # print(target[mask_obj][:,5:])
    # print(torch.argmax(target[mask_obj][:,5:],dim=1))
    # cls_loss = cls_loss_func(output[mask_obj][:, 5:], torch.argmax(target[mask_obj][:, 5:], dim=1))
    # 分类标签没有使用one-hot
    # 需要把标签也转为Tensor float32的数据类型
    cls_loss = cls_loss_func(output[mask_obj][:, 5:], (target[mask_obj][:, 5] - 1).long())
    loss = alpha * c_loss + (1 - alpha) * (off_loss + cls_loss)
    return loss


if __name__ == '__main__':
    if os.path.exists("log"):
        shutil.rmtree("log")
        print("log off")
    myDataset = dataset.MyDataset(cfg.train_path, cfg.img_path)
    train_loader = DataLoader(myDataset, batch_size=32, shuffle=True)
    writer = SummaryWriter("log")
    # net = Darknet53().cuda()
    net = YoloBody(cfg.ANCHORS_GROUP1, 2, 'n').cuda()
    # 判断是否有权重，有就加载权重
    # 局限性：不适用于给一个异常的预权重且名字为param_path+ "_old"的情况
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
        try:
            net.load_state_dict(torch.load(param_path + "_old"))
            os.rename(param_path + "_old", param_path)
            print("Only old param is loaded")
        except RuntimeError as r:
            print(r)
            print("参数错误，将开始重新训练")
    else:
        print("NO Param!")

    net.train()
    opt = torch.optim.Adam(net.parameters())
    # opt = torch.optim.SGD(net.parameters(), lr=0.001)
    count = 1
    while count <= 500:
        sum_loss = 0.
        sum_target = 0.
        pre_target = 0.
        for target_13, target_26, target_52, img_data in tqdm(train_loader, file=sys.stdout):
            output_13, output_26, output_52 = net(img_data.cuda())
            # # 全部放到cuda上运算
            # target_13 = target_13.cuda()
            # target_26 = target_26.cuda()
            # target_52 = target_52.cuda()
            # # 计算标签目标总数
            # t13 = _target_cal(target_13, is_reshape=False)
            # sum_target += t13 * 3
            # # 对输出进行变形，以及对输出的两个分类概率取argmax
            # o13 = _target_cal(output_13)
            # o26 = _target_cal(output_26)
            # o52 = _target_cal(output_52)
            # # 和标签比较
            # r13 = o13.eq(target_13[..., -1] + 1)
            # r26 = o26.eq(target_26[..., -1] + 1)
            # r52 = o52.eq(target_52[..., -1] + 1)
            # # 统计输出中分类正确的预测的目标数
            # pre_target += (_target_cal(r13, False) + _target_cal(r26, False) + _target_cal(r52, False))
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
        # print("sum_target: ", sum_target)
        # avg_acc = pre_target / sum_target
        avg_loss = sum_loss / len(train_loader)
        writer.add_scalar("loss", avg_loss, count)
        print("epoch {} avg loss is: {}".format(count, avg_loss))
        # print("epoch {} avg acc is: {}".format(count, avg_acc))
        if os.path.exists(param_path):
            os.rename(param_path, param_path + "_old")
        # print(param_path)
        torch.save(net.state_dict(), param_path)
        print("save success!")
        if os.path.exists(param_path + "_old"):
            os.remove(param_path + "_old")
        # print("delete old param success")
        count += 1
