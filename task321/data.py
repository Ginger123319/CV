from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np


# 定义数据集操作类，继承自torch.utils.data包中的Dataset类
class MnistDataset(Dataset):
    def __init__(self, root, is_train=True):
        super(MnistDataset, self).__init__()
        # 定义一个接收数据的列表，用来接收图片转化为数组后的数据
        self.dataset = []
        # 对图片路径img_path以及标签tag【0~9】进行处理
        # 最终将图片路径与标签对应起来，在列表中展示为多个元组
        sub_dir = "Train" if is_train else "Test"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path, tag))

    # 返回列表的长度
    def __len__(self):
        return len(self.dataset)

    # 获取列表中的单条数据
    def __getitem__(self, index):
        data = self.dataset[index]
        # 使用opencv读取图片,并转为灰度图，color为单通道，形状为(28,28,1)
        img_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
        # 一张图片就是这28x28X1共784个像素点，就是一个输入，通过这784个像素点就得到了该图片的所有信息
        # 而三通道的图片不额能这样处理？？
        img_data = img_data.reshape(28 * 28 * 1)
        # 由于存在较大的元素，所以在后续计算中容易超出数据类型的精度，所以需要进行归一化处理
        # 将0~255的数字 处理在0~1之间；还有就是减小模型中的数据运算量提升性能
        img_data = img_data / 255

        # one-hot
        # 取出来的tag是多少，就在这个张量中对应索引位置处设为1
        tag_one_hot = np.zeros(10)
        tag_one_hot[int(data[1])] = 1

        return np.float32(img_data), np.float32(tag_one_hot)


if __name__ == '__main__':
    dataset = MnistDataset(r"D:\Python\code\source\MNIST_IMG")
    print(dataset.dataset)
    dataLoader = DataLoader(dataset, batch_size=512, shuffle=True)
    for i, (x, y) in enumerate(dataLoader):
        print(i)
        print(x.shape)
        print(y.shape)
