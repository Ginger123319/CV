# 将yolo格式标注的txt文件转化为coco数据集标注格式的json文件类型
# yolo格式为(xc,yc,w,h)相对坐标  coco标注格式为(xmin,ymin,w,h),绝对坐标
# voc标注xml格式为(xmin,ymin,xmax,ymax)
import os
import sys
import cv2
import json
import shutil
import argparse
from datetime import datetime

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
images_set = set()
image_id = 000000
annotation_id = 0


def addCatItem(categroy_dict):  # 保存所有的类别信息
    for k, v in categroy_dict.items():
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(k)
        category_item['name'] = v
        coco['categories'].append(category_item)


def addImgItem(file_name, size):
    global image_id
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size[1]
    image_item['height'] = size[0]
    image_item['license'] = None
    image_item['flickr_url'] = None
    image_item['coco_url'] = None
    image_item['data_captured'] = str(datetime.today())
    coco['images'].append(image_item)
    images_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox  is  x,y,w,h    seg.append(bbox[0])    seg.append(bbox[1])
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
    annotation_item['segmentation'].append(seg)
    annotation_item['area'] = bbox[2] * bbox[3]  # w*h
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def xywhn2xywh(bbox, size):  # 从yolo标注到coco标注
    bbox = list(map(float, bbox))
    size = list(map(float, size))  # h,w
    xmin = (bbox[0] - bbox[2] / 2) * size[1]
    ymin = (bbox[1] - bbox[3] / 2) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    bbox = (xmin, ymin, w, h)
    return list(map(int, bbox))


def parseXmlFilse(image_path, anno_path, save_path, json_name):
    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_path = os.path.join(save_path, json_name)
    category_set = []
    with open(anno_path + '/classes.txt', 'r') as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_id = dict((k, v) for k, v in enumerate(category_set))
    addCatItem(category_id)
    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path)]
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    for file in files:
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img = cv2.imread(images[index])
            shape = img.shape
            filename = images[index].split(os.sep)[-1]
            current_image_id = addImgItem(filename, shape)
        else:
            continue
        with open(file, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split()
                category = int(i[0])
                category_name = category_id[category]
                bbox = xywhn2xywh((i[1], i[2], i[3], i[4]), shape)
                addAnnoItem(category_name, current_image_id, category, bbox)

    json.dump(coco, open(json_path, 'w'))
    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("bbox nums:{}".format(len(coco['annotations'])))


if __name__ == '__main__':
    '''参数说明：
            anno_path:标注txt文件存储地址
            save_path:json文件输出文件夹
            image_path:图片路径
            json_name:保存json文件名称'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='/home/train_data_v5_format/labels',
                        help='yolo txt path')
    parser.add_argument('-s', '--save-path', type=str, default='/home/train_data_v5_format/anno_json',
                        help='json save path')
    parser.add_argument('--image-path', default='/home/train_data_v5_format/images/train')
    parser.add_argument('--json-name', default='train.json')

    opt = parser.parse_args()
    if len(sys.argv) > 1:
        print(opt)
        parseXmlFilse(**vars(opt))
    else:
        anno_path = r'D:\download\archive\ships-aerial-images\valid\labels'
        save_path = r'D:\download\archive\ships-aerial-images\valid'
        image_path = r'D:\download\archive\ships-aerial-images\valid\images'
        json_name = r'annotation.json'
        parseXmlFilse(image_path, anno_path, save_path, json_name)
