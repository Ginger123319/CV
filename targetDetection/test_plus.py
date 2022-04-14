import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataDeal import MyData
from net_plus import Net
from utils import iou
from PIL import Image, ImageDraw

params_path = "../../source/target_detection/param_plus"

# 导入测试集
test_data = MyData("../../source/target_detection", is_train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

net = Net()
if os.path.exists(params_path):
    net.load_state_dict(torch.load(params_path))
    print("参数加载成功!")
else:
    print("No param!")
# 开启测试模式
net.eval()

num_no_target = 0
num_target = 0
pic_flag = True

for i, (img, label, label2) in enumerate(test_loader):
    print(f"第{i + 1}次测试")
    out, out2 = net(img)
    # print(img.shape)
    # print(label.shape)
    # print(label2.shape)
    # print(out.shape)
    # print(out2)
    # break
    # result = iou(out.detach().numpy(), label.detach().numpy())
    # print(f"test_iou is {result}")

    # 注意index为元组类型，需要取出第一个元素进行判断是否有值
    out2 = out2.detach().cpu()
    label2 = label2.detach().cpu()

    out = out.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    # index = torch.where(out2 > 0.5)
    # print(index)
    # print(index[0])
    print("out2 is {}".format(out2[0][0]))
    if out2[0][0] < 0.5:
        avg_iou = 0
        num_no_target += 1
        pic_flag = False
    else:
        num_target += 1
        avg_iou = iou(out, label)
        pic_flag = True
    print(f"test_iou is {avg_iou}")
    # 判断为有小黄人的图片都进行画框操作
    if pic_flag:
        # 开始画框【此处操作不熟悉，比如多维张量的索引取值label[0]】
        out = out * 300
        label = label * 300
        img = img.permute(0, 2, 3, 1)
        img_data = np.array(img[0] * 255, dtype=np.uint8)
        img = Image.fromarray(img_data, "RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(np.array(label[0]), outline="red", width=2)
        draw.rectangle(np.array(out[0]), outline="yellow", width=2)
        # img.show()
        img.save("../../source/target_detection/result_pic_plus/{}.png".format(i + 1))
print("没有小黄人和有小黄人的图片数目为{0} {1}".format(num_no_target,num_target))
print("图片结果全部已经保存！")
