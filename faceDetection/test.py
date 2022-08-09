import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataDeal import MyData
from net import Net
from utils import iou
from PIL import Image, ImageDraw

params_path = r"D:\Python\source\FACE\param"

# 导入测试集
test_data = MyData(r"D:\Python\source\FACE\celebA", is_train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

net = Net()
if os.path.exists(params_path):
    net.load_state_dict(torch.load(params_path))
    print("参数加载成功!")
else:
    print("No param!")
# 开启测试模式
net.eval()
for i, (img, label) in enumerate(test_loader):
    print(f"第{i}次测试")
    out = net(img)
    # print(img.shape)
    # print(label.shape)
    result = iou(out.detach().numpy(), label.detach().numpy())
    print(f"test_iou is {result}")
    # 开始画框【此处操作不熟悉，比如多维张量的索引取值label[0]】
    out = out.detach().numpy() * 300
    label = label.detach().numpy() * 300
    img = img.permute(0, 2, 3, 1)
    img_data = np.array(img[0] * 255, dtype=np.uint8)
    img = Image.fromarray(img_data, "RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(np.array(label[0]), outline="red", width=2)
    draw.rectangle(np.array(out[0]), outline="yellow", width=2)
    # img.show()
    img.save(r"D:\Python\source\FACE\test_result/{}.png".format(i + 1))
    if i >= 99:
        break
print("图片结果全部已经保存！")
