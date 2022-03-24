from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# 下载数据集
# 导入数据集，处理数据集中的data和targets
# transforms.ToTensor()只是对数据进行基本的转换
# 从PILImage转为tensor类型
# 归一化处理，数据处理为0到1之间的浮点数
# 换轴，从HWC->CHW
# 不过只有在加载数据之后的调用过程中才会对图片进行实时处理
train_dataset = datasets.MNIST(root=r"..\..\source\MNIST_DATA", train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root=r"..\..\source\MNIST_DATA", train=False, download=False,
                              transform=transforms.ToTensor())

print(train_dataset)
print(test_dataset)
print(train_dataset.data.shape)
# 打印标签的形状
print(train_dataset.targets.shape)
print(train_dataset.data[0])
print(train_dataset.targets[:10])
# 展示索引为1的图片
# img = Image.fromarray(np.array(train_dataset.data[1]),"L")
# img.show()
# 直接将tensor或者np数组转换为PILImage类型，可以直接进行显示
# unloader = transforms.ToPILImage()
# img = unloader(train_dataset.data[0])
# img.show()

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
for i, (img, label) in enumerate(train_loader):
    print(img)
    print(label)
    print(img.shape)
    img = img.reshape(-1, 28 * 28)
    print(img.shape)
