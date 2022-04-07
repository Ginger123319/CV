# 60000张图片分成10个类别放在不同的文件夹中
# 取到文件夹中的数据和标签（目录名称0~9）
# 需要使用torch中处理数据集的工具包
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


# 定义一个数据处理类
class MnistDataset(Dataset):
    # 给一个文件存储的根路径，以及是否是训练集的一个布尔判断标志
    def __init__(self, root, is_train=True):
        super(MnistDataset, self).__init__()
        # 定义一个属性，空列表存储数据
        self.dataset = []
        # 判别是否是训练集，来确定子目录的名字
        sub_dir = "TRAIN" if is_train else "TEST"
        # 导入os模块，遍历子目录（0~9），将标签保留
        for tag in os.listdir(f"{root}/{sub_dir}"):

            # 此处只存储图片路径，直接存储图片占用空间过大，只需要在使用的时候读取出来就行
            for img_name in os.listdir(f"{root}/{sub_dir}/{tag}"):
                img_path = f"{root}/{sub_dir}/{tag}/{img_name}"
                # 将数据和标签一一对应组成一个元组添加到列表中供后续处理
                self.dataset.append((img_path, tag))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 处理列表中的一个元组，一个是图片路径一个是tags
        # 处理图片为numpy数组或者tensor张量
        elem = self.dataset[index]
        # 读出来就是一个np的数组28*28*1,强转灰度图(增加参数：cv2.IMREAD_GRAYSCALE)
        img_data1 = cv2.imread(elem[0], cv2.IMREAD_GRAYSCALE).reshape(-1)
        # 对数据做归一化处理
        img_data = img_data1 / 255
        # 处理tags，采用one-hot编码方式
        # 十分类问题，创建10个元素的向量
        tag_one_hot = np.zeros(10)
        # tag在elem元组中是一个字符串，需要转换一下
        tag_one_hot[int(elem[1])] = 1
        # 返回的时候将所有数据转为float32类型
        return np.float32(img_data), np.float32(tag_one_hot)


if __name__ == '__main__':
    data = MnistDataset(r"..\..\source\MNIST_IMG", True)
    print(data[0])
