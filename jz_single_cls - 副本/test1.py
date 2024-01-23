# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import os
# import torch
# from pathlib import Path
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# from torch.utils.data import DataLoader
#
# from dc_model_repo import model_repo_client
# import torchvision.transforms as transforms
#
# # put your code here
# dc.logger.info("ENV is ready!")
#
# # 构建工作路径以及创建文件夹
# work_dir = model_repo_client.get_dc_proxy().get_work_dir()
# work_dir = os.path.join(work_dir, dc.conf.global_params.block_id, "work_files")
# Path(work_dir).mkdir(parents=True, exist_ok=True)
# print("Current work_dir: {}".format(work_dir))
#
#
# def feature_reshape(feature_map):
#     n, c, h, w = feature_map.shape
#     feature = feature_map.reshape(n, c, h * w).permute(2, 0, 1)
#     return feature, n, c, h, w
#
#
# def feature_recovery(feature_map, n, c, h, w):
#     feature = feature_map.permute(1, 2, 0).reshape(n, c, h, w)
#     return feature
#
#
# # 定义数据预处理
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
#
# # 加载数据集
#
# trainset = torchvision.datasets.CIFAR10(root=os.path.join(work_dir, 'data'), train=True, download=True,
#                                         transform=transform_train)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root=os.path.join(work_dir, 'data'), train=False, download=True,
#                                        transform=transform_test)
# testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#
#
# # 定义模型
# class ViTResNet50(nn.Module):
#     def __init__(self, num_classes=10, pretrained=True):
#         super(ViTResNet50, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=pretrained)
#
#         self.vit1 = nn.TransformerEncoder(nn.TransformerEncoderLayer(
#             d_model=256, nhead=8), num_layers=4)
#         self.vit2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(
#             d_model=512, nhead=8), num_layers=4)
#         self.vit3 = nn.TransformerEncoder(nn.TransformerEncoderLayer(
#             d_model=1024, nhead=8), num_layers=4)
#         self.vit4 = nn.TransformerEncoder(nn.TransformerEncoderLayer(
#             d_model=2048, nhead=8), num_layers=4)
#         self.conv = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#
#         x = self.resnet.layer1(x)
#         x, n, c, h, w = feature_reshape(x)
#         x = self.vit1(x)
#         x = feature_recovery(x, n, c, h, w)
#
#         x = self.resnet.layer2(x)
#         x, n, c, h, w = feature_reshape(x)
#         x = self.vit2(x)
#         x = feature_recovery(x, n, c, h, w)
#
#         x = self.resnet.layer3(x)
#         x, n, c, h, w = feature_reshape(x)
#         x = self.vit3(x)
#         x = feature_recovery(x, n, c, h, w)
#
#         x = self.resnet.layer4(x)
#         x, n, c, h, w = feature_reshape(x)
#         x = self.vit4(x)
#         x = feature_recovery(x, n, c, h, w)
#
#         x = self.conv(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         return x
#
#
# # 定义训练函数
# def train(model, trainloader, optimizer, criterion, device):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#
#     acc = 100. * correct / total
#     return train_loss / len(trainloader), acc
#
#
# # 定义测试函数
# def test(model, testloader, criterion, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#     acc = 100. * correct / total
#     return test_loss / len(testloader), acc
#
#
# # 训练
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = ViTResNet50().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
#
# for epoch in range(350):
#     train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
#     test_loss, test_acc = test(model, testloader, criterion, device)
#     scheduler.step()
#
#     print('Epoch: %d, Train Loss: %.4f, Train Acc: %.2f, Test Loss: %.4f, Test Acc: %.2f' % (
#         epoch, train_loss, train_acc, test_loss, test_acc))
#
# # 测试模型
# test_loss, test_acc = test(model, testloader, criterion, device)
# print('Test Loss: %.4f, Test Acc: %.2f' % (test_loss, test_acc))
#
# dc.logger.info("done.")
# import torch
# import torchvision.models as models
#
# print(torch.cuda.is_available())
# s1 = "a \' \" \"\"\"(1.jpg"
# s1 = "a b 1 2 3 4 5"
# print(" ".join(s1.split()[-6::-1][::-1]))
# exit()
# print(s1)
# print(repr(s1))
# print(s1 == 'a \' " """(1.jpg')
# exit()
#
# # 定义ResNet50模型
# model = models.resnet50(pretrained=False)
#
# # 设置输入图片的尺寸和数据类型
# input_size = (32, 3, 224, 224)
# input_dtype = torch.float32
#
# # 计算每个像素所需的字节数
# if input_dtype == torch.uint8:
#     pixel_size = 1
# elif input_dtype == torch.float16:
#     pixel_size = 2
# else:
#     pixel_size = 4
#
# # 计算模型所需的内存
# model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # 以MB为单位
# input_size_mb = torch.tensor(input_size).prod() * pixel_size / (1024 ** 2)  # 以MB为单位
# gpu_memory = (model_size + input_size_mb) * 2  # 乘以2是为了考虑反向传播时的内存占用
# print((32 * (224 * 224 * 3 * pixel_size) * 4 / (1024 ** 2)) * model_size * 1.8)
#
# print("ResNet50模型大小：", model_size, "MB")
# print("输入图片大小：", input_size_mb, "MB")
# print("所需GPU内存：", gpu_memory, "MB")
#
# import cv2
#
# img = cv2.imread(r'D:\Python\test_jzyj\key_mouse_20\和\0001.jpg')
# print(img)
# img = cv2.imread(r'D:\Python\test_jzyj\key_mouse_20\pic\0002.jpg')
# print(img.shape, img.dtype)
# print(img.shape[0] * img.shape[1] * 3 * 1 / 1024)
#
#
# def create_classes_txt(classes, work_dir):
#     # 生成class.txt文件，用于记录目标检测的类别
#     f = open(work_dir + '/middle_dir/data_info/classes.txt', 'w')
#     for i in classes:
#         f.write(i)
#         f.write('\n')
#     f.close()
#     print("classes.txt has been created.")
#
#
# def file_lines_to_list(path):
#     # open txt file lines to a list
#     with open(path) as f:
#         content = f.readlines()
#     # remove whitespace characters like `\n` at the end of each line
#     content = [x.strip() for x in content]
#     return content
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets

# 定义数据预处理和增强的方法

train_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.AutoAugment(),
    transforms.ToTensor()
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=r'D:\download\DatasetId_1791208_1679636382\DatasetId_1791208_1679636382\img',
                                     transform=train_transform)
import numpy as np

# 定义二维向量
vec = np.random.rand(3, 1)
print(type(vec))
print(vec)
index = [0, 1]
tmp_vec = vec[index]
vec[index][:, 0] = [0, 0]
print(vec)
exit()
# 定义32个不同的值
values = np.arange(0, 32, 1)
print(values)

# 将第二个维度的第2个位置全部值按顺序替换为不同的32个值
vec[:, 1] = values

# 输出替换后的向量
print(vec)
exit()
# 显示图片
img, label = train_dataset[200]
print(img.dtype)
img = img.permute(1, 2, 0)
print(img.shape)

plt.imshow(img)
plt.show()
