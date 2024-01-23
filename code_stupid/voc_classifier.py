# 从VOC数据集中把分类用的图片提取出来
"""
这80个文件中每一行的图像ID后面还跟了一个数字，要么是-1， 要么是1，有时候也可能会出现0，
意义为：-1表示当前图像中，没有该类物体；1表示当前图像中有该类物体；0表示当前图像中，该类物体只露出了一部分
"""
import os
import shutil
from tqdm import tqdm

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

img_path = r'D:\Python\source\VOCdevkit\VOC2012\JPEGImages'
class_path = r'D:\Python\source\VOCdevkit\VOC2012\ImageSets\Main'
save_path = r'D:\Python\source\VOCdevkit\VOC2012\val'

for c in tqdm(classes):
    txt_path = os.path.join(class_path, c + '_val.txt')
    print(txt_path)

    move_image = os.path.join(save_path, c)
    print(move_image)

    if not os.path.exists(move_image):
        os.makedirs(move_image)

    with open(txt_path, 'r') as f:
        img_files = f.readlines()

    for name in img_files:
        img_name = name.split(' ')[0]

        src_img = os.path.join(img_path, img_name + '.jpg')
        target_img = os.path.join(move_image, img_name + '.jpg')
        shutil.copy(src_img, target_img)
        exit()
