# coding=gbk
import shutil
import torch
import os
from torchvision.utils import save_image
from model.data_eye import DataEye
from model.u2net import U2NET
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter

# �����洢·��
params_path = r"..\..\..\source\EYE_DATA\params_path"

# ����ǰɾ��SummaryWriter���ɵ�logĿ¼
if os.path.exists(r"./log"):
    shutil.rmtree(r"./log")
    print("log is deleted��")

# ʵ����SummaryWriter
summaryWriter = SummaryWriter(r"./log")

# ����ѵ����
train_data = DataEye(r"..\..\..\source\EYE_DATA", is_train=True)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

# ������Լ�
test_data = DataEye(r"..\..\..\source\EYE_DATA", is_train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# ����ģ�����ʵ����
net = U2NET()
opt = optim.Adam(net.parameters())
loss_func = BCELoss()

# ����ԤȨ��
if os.path.exists(params_path):
    net.load_state_dict(torch.load(params_path))
    print("�������سɹ���")
else:
    print("No params_path!")


# ������ʧ����
def multi_loss_func(r0, r1, r2, r3, r4, r5, r6, labels):
    loss0 = loss_func(r0, labels)
    loss1 = loss_func(r1, labels)
    loss2 = loss_func(r2, labels)
    loss3 = loss_func(r3, labels)
    loss4 = loss_func(r4, labels)
    loss5 = loss_func(r5, labels)
    loss6 = loss_func(r6, labels)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


# ��ʼѵ��
labels_v = torch.zeros(4)
out = torch.zeros(4)
for epoch in range(100):
    # ÿһ������һ��loss�ۼ�ֵ
    sum_loss = 0.
    last_loss = 0.
    # ����ѵ��ģʽ
    net.train()
    for i, (img, tag) in enumerate(train_loader):
        print(f"��{i}��")
        d0, d1, d2, d3, d4, d5, d6 = net(img)
        out = d0
        labels_v = tag
        # print(d0.shape)
        # print(labels_v.shape)
        final_loss, total_loss = multi_loss_func(d0, d1, d2, d3, d4, d5, d6, labels_v)

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        sum_loss += total_loss.item()
        last_loss += final_loss.item()

    avg_sum_loss = sum_loss / len(train_loader)
    avg_last_loss = last_loss / len(train_loader)

    summaryWriter.add_scalars("loss", {"avg_sum_loss": avg_sum_loss, "avg_last_loss": avg_last_loss}, epoch + 1)
    print("SummaryWriter��ӳɹ���")

    print(f"��{epoch}�֣�avg_sum_loss is {avg_sum_loss}")
    print(f"��{epoch}�֣�avg_last_loss is {avg_last_loss}")

    # ����ͼƬ
    save_image(out, r"..\..\..\source\EYE_DATA\savedImg\fake_img{}.png".format(epoch), nrow=1)
    save_image(labels_v, r"..\..\..\source\EYE_DATA\savedImg\real_img{}.png".format(epoch), nrow=1)

    # ÿһ�ֱ���һ��ѵ������
    torch.save(net.state_dict(), params_path)
    print("��������ɹ���")

    # ��������ģʽ
    net.eval()
    for i, test_img in enumerate(test_loader):
        print(f"��{i}�β��ԣ�")
        d0, d1, d2, d3, d4, d5, d6 = net(test_img)
        avg = torch.mean(d0)
        out = (d0 > avg).float()
        save_image(out, r"..\..\..\source\EYE_DATA\savedImg\test_result_img{}.png".format(i), nrow=1)
        save_image(test_img, r"..\..\..\source\EYE_DATA\savedImg\test_real_img{}.png".format(i), nrow=1)
