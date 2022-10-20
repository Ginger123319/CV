import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 只要能返回数据以及对应的标签即可
from torchvision.utils import save_image


class MyData(Dataset):
    def __init__(self, root):
        super(MyData, self).__init__()
        self.path = root
        # 拿到当前目录下的所有文件名称
        self.tag_names = os.listdir(os.path.join(root, "SegmentationClass"))
        self.data_names = os.listdir(os.path.join(root, "JPEGImages"))

    def __len__(self):
        return len(self.tag_names)

    def __getitem__(self, index):
        # 生成黑色背景
        # 此处的函数调用像forward函数一样，在调用时可以省略函数名字
        # 在类中被定义为一个call方法，调用的时候可以不写函数名，直接传入参数
        # def __call__(self, pic):
        #     return F.to_pil_image(pic, self.mode)
        black0 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        black1 = transforms.ToPILImage()(torch.zeros(3, 256, 256))
        tag_name = self.tag_names[index]
        data_name = tag_name[:-3] + "jpg"
        tag_path = os.path.join(self.path, "SegmentationClass")
        data_path = os.path.join(self.path, "JPEGImages")
        # 打开两张图片
        # mode=P?
        img_tag = Image.open(os.path.join(tag_path, tag_name))
        img_data = Image.open(os.path.join(data_path, data_name))

        img_size = torch.Tensor(img_tag.size)

        l_max_index = img_size.max()
        ratio = 256 / l_max_index
        img_resize = (img_size * ratio).long()

        img_tag_resize = img_tag.resize(img_resize)
        img_data_resize = img_data.resize(img_resize)

        black0.paste(img_data_resize)
        black1.paste(img_tag_resize)

        # 转为Tensor，换轴以及归一化处理
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(black0), transform(black1)


if __name__ == '__main__':
    dataset = MyData(r"..\..\source\VOCdevkit\VOC2012")
    # print(dataset.names)
    # print(type(dataset))
    print(dataset[0])
    # i = 1
    # for a, b in dataset:
    #     print(i)
    #     print(a.shape)
    #     print(b.shape)
    #     save_image(a, "data/{0}.jpg".format(i), nrow=1)
    #     save_image(b, "data/{0}.png".format(i), nrow=1)
    #     i += 1
