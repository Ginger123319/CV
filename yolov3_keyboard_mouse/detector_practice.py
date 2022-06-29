import torch
import tool
import os
import cfg
import xml_reader
import torchvision
from module import *
import matplotlib.pyplot as plt
from PIL import Image, ImageFont
import PIL.ImageDraw as draw

from utils import pic_resize

device = "cuda" if torch.cuda.is_available() else "cpu"
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


class Detector(torch.nn.Module):  # 定义侦测模块

    def __init__(self):
        super().__init__()

        self.net = Darknet53()  # 实例化网络
        self.net.load_state_dict(torch.load(cfg.param_path))  # 加载网络参数
        self.net.eval()  # 固化参数即使得BatchNormal不在测试的时候起作用

    def forward(self, input, thresh, anchors, ratio):  # 定义前向运算，并给三个参数分别是输入数据，置信度阀值，及建议框
        output_13, output_26, output_52 = self.net(input)  # 将数据传入网络并获得输出
        # （n,h,w,3,15）其中n,h,w,3做为索引，即这里的idxs_13，表示定义在那个格子上
        idxs_13, vecs_13 = self._filter(output_13, thresh)  # 赛选获取13*13特侦图置信度合格的置信度的索引和15个值，赛选函数下面自己定义的
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13], ratio)  # 对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        idxs_26, vecs_26 = self._filter(output_26, thresh)  # 赛选获取26*26特侦图置信度合格的置信度的索引和15个值，赛选函数下面自己定义的
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26], ratio)  # 对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        idxs_52, vecs_52 = self._filter(output_52, thresh)  # 赛选获取52*52特侦图置信度合格的置信度的索引和15个值，赛选函数下面自己定义的
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52], ratio)  # 对上面输出的两个值进行解析，获得最终的输出框，解析函数在下面
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)  # 将三种框在0维度按顺序拼接在一起并返回给调用方

    def _filter(self, output, thresh):  # 过滤置信度函数，将置信度合格留下来
        output = output.permute(0, 2, 3, 1)  # 数据形状[N,C,H,W]-->>标签形状[N,H,W,C]，，因此这里通过换轴

        # 通过reshape变换形状 [N,H,W,C]即[N,H,W,45]-->>[N,H,W,3,15]，分成3个建议框每个建议框有15个值
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = output[..., 0] > thresh  # 获取输出置信度大于置信度阀值的目标值的掩码（即布尔值）
        # print(mask)
        # exit()
        # print(output.shape)
        # print(mask.shape)
        # exit()
        idxs = mask.nonzero()  # 将索引取出来其形状N,V包含（N,H,W,3）
        vecs = output[mask]  # 通过掩码获取置信度大于阈值的对应数据【iou,cx,cy,w,h,cls】长度为5+10的向量
        # print(vecs.shape)
        # exit()
        return idxs, vecs  # 将索引和数据返回给调用方

    def _parse(self, idxs, vecs, t, anchors, ratio):  # 定义解析函数，并给4个参数分别是上面筛选合格的框的索引，9个值（中心点偏移和框的偏移即类别数），
        # t是每个格子的大小，t=总图大小/特征图大小，anchors建议框
        anchors = torch.Tensor(anchors)  # 将建议框转为Tensor类型
        # idx形状NV；V中包含这些信息（n，h，w，3）
        a = idxs[:, 3]  # 表示拿到3个框对应的索引
        confidence = torch.sigmoid(vecs[:, 0])  # 网络输出没有做sigmoid处理，因此值可能大于1，但是期望输出是一个概率，应该在0~1之间
        # confidence = vecs[:, 0]  # 获取置信度vecs里面有5+类别数个元素，第一个为置信度，因此取所有的第0个

        _classify = vecs[:, 5:]  # 获取分类的类别数据

        if len(_classify) == 0:  # 判断类别数的长的是否为0为0返回空，避免代码报错
            classify = torch.Tensor([])
        else:
            classify = torch.argmax(_classify, dim=1).float()  # 如果不为0，返回类别最大值的索引，这个索引就代表类别
        # print("cls:", classify.shape)
        # idx形状（n，h，w，3），vecs（iou，p_x，p_y，p_w，p_h，类别）这里p_x，p_y代表中心点偏移量,p_w，p_h框偏移量
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 计算中心点cy（h+p_y)
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 计算中心点cx（h+p_x)
        # a = idxs[:, 3]  # 表示拿到3个对应的索引
        w = anchors[a, 0] * torch.exp(vecs[:, 3])  # 计算实际框的宽为w,w=建议框的宽*框的偏移量p_w
        h = anchors[a, 1] * torch.exp(vecs[:, 4])  # 计算实际框的高为h,h=建议框的高*框的偏移量p_h
        # 还原为在原图上的中心点和宽高
        cx /= ratio
        cy /= ratio
        w /= ratio
        h /= ratio

        x1 = cx - w / 2  # 计算框左上角x的坐标
        y1 = cy - h / 2  # 计算框左上角y的坐标
        x2 = x1 + w  # 计算框右下角x的坐标
        y2 = y1 + h  # 计算框右下角y的坐标
        out = torch.stack([confidence, x1, y1, x2, y2, classify], dim=1)  # 将置信度坐标和类别按照一轴即列的方向重组堆叠在一起
        return out


if __name__ == '__main__':
    # 在原图上画框，需要将输出的坐标值和宽高除以缩放比例，才是在原图上的中心点坐标和宽高
    # 因为kmeans中产生的建议框也是缩放为416*416尺寸后的，所以输出的宽高也是416*416图片上的宽高，所以需要还原
    # 标签是缩放后的标签，因此得到的输出值需要还原到原图上

    detector = Detector()  # 实例化侦测模块
    # img_path = 'data/img_after/'
    img_path = cfg.img_path
    img_name_list = []
    with open(cfg.val_path) as f:
        for line in f.readlines():
            img_name_list.append(line.split()[0])
    name = xml_reader.class_name
    color = xml_reader.color_name
    font = ImageFont.truetype("simsun.ttc", 25, encoding="unic")  # 设置字体
    for image_file in img_name_list:
        im = Image.open(os.path.join(img_path, image_file))
        _img_data, ratio = pic_resize(os.path.join(img_path, image_file))
        img_data = transforms(_img_data)
        img = img_data[None, :]
        out_value = detector(img, 0.25, cfg.ANCHORS_GROUP, ratio)  # 调用侦测函数并将数据，置信度阀值和建议框传入
        # print(out_value.shape)
        # exit()
        boxes = []  # 定义空列表用来装框

        for j in range(2):  # 循环判断类别数
            # 获取同一个类别的掩码
            # 输出的类别如果和类别相同就做nms删掉iou大于阈值的框留下iou小的表示这不是同一个物体
            classify_mask = (out_value[..., -1] == j)

            _boxes = out_value[classify_mask]  # 取出所有同类别的框
            boxes.append(tool.nms(_boxes))  # 对同一类别做nms删掉不合格的框，将一个类别的框放在一起添加进列表
        for box in boxes:  # 遍历各个类别
            try:
                # print(box.shape)
                # exit()
                img_draw = draw.ImageDraw(im)  # 制作画笔
                for i in range(len(box)):  # 遍历各个类别的所有框并进行画图
                    c, x1, y1, x2, y2, cls = box[i, :]  # 将自信度和坐标及类别分别解包出来
                    # print(c,x1, y1, x2, y2)
                    # print(int(cls.item()))
                    # print(round(c.item(),4))#取值并保留小数点后4位
                    img_draw.rectangle((x1, y1, x2, y2), outline=color[int(cls.item())], width=4)  # 画框

                    img_draw.text((max(x1, 0) + 3, max(y1, 0) + 5), fill=color[int(cls.item())],
                                  text=str(int(cls.item())), font=font, width=2)
                    img_draw.text((max(x1, 0) + 20, max(y1, 0) + 5), fill=color[int(cls.item())],
                                  text=name[int(cls.item())], font=font, width=2)
                    img_draw.text((max(x1, 0) + 3, max(y1, 0) + 30), fill=color[int(cls.item())],
                                  text=str(round(c.item(), 4)), font=font, width=2)
            except:
                continue
        # im.save(os.path.join('result', image_file))
        plt.clf()
        plt.ion()
        plt.axis('off')
        plt.imshow(im)
        # plt.show()
        plt.pause(3)
        plt.close()
        # im.show()

'============================================================================================================='
'''
nonzero(a) 
nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。它的返回值是一个长度为a.ndim(数组a的轴数)的元组，元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。

（1）只有a中非零元素才会有索引值，那些零值元素没有索引值；

（2）返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。

（3）索引值数组的每一个array均是从一个维度上来描述其索引值。比如，如果a是一个二维数组，则索引值数组有两个array，第一个array从行维度来描述索引值；第二个array从列维度来描述索引值。

（4）transpose(np.nonzero(x))函数能够描述出每一个非零元素在不同维度的索引值。

（5）通过a[nonzero(a)]得到所有a中的非零值
'''
