import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test', 'val']
# 调整自己的数据集类别
classes = ["键盘", "鼠标"]


# 运行 03.voc_label.py 代码（需要手动修改代码里面的目标类别），将 VOC2007/Annotations 路径下的标签，
# 生成一个文件夹 VOC2007/labels，里面保存的是每张图片的标签坐标（以 txt格式保存）

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('./VOC2007/Annotations/%s.xml' % image_id, encoding='utf-8')
    out_file = open('./voc2007/labels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('./VOC2007/labels/'):
        os.makedirs('./VOC2007/labels/')
    image_ids = open('./VOC2007/ImageSets/Main/%s.txt' % image_set).read().strip().split()
    list_file = open('./VOC2007/%s.txt' % image_set, 'w')
    for image_id in image_ids:
        list_file.write('./VOC2007/Images/%s.jpg\n' % image_id)
        convert_annotation(image_id)
    list_file.close()
