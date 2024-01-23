import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from .aug_util import get_random_data



MEANS = (104, 117, 123)
class SSDDataset(Dataset):
    def __init__(self, train_lines, image_size, aug_config=None, mosaic=False):
        super(SSDDataset, self).__init__()

        self.mosaic = mosaic
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.aug_config = aug_config

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def load_index_img(self, index):
        lines = self.train_lines
        annotation_line=lines[index]
        annos = annotation_line.split()
        image = Image.open(annos[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in annos[1:]])

        from pathlib import Path
        img, y = get_random_data(image=image, box=box, input_shape=self.image_size[0:2], aug_config=self.aug_config, index=Path(annos[0]).name)
        
        # self.draw_dir="/home/aps/zk/pba-ssd/output/draws"
        # from PIL import ImageDraw
        # img_draw = Image.fromarray(img.astype(np.uint8))
        # d = ImageDraw.Draw(img_draw)
        # colors = ["red", "green", "blue", "pink", "orange"]  # list(ImageColor.colormap.keys())
        # for i, box in enumerate(y):
        #     color = colors[i % len(colors)]
        #     d.rectangle(box[:-1].tolist(), outline=color, width=3)
        #     d.text(box[:2].tolist(), str(y[-1]), fill=color)
        # draw_dir = Path(self.draw_dir)
        # if not draw_dir.exists():
        #     draw_dir.mkdir(parents=True)
        # img_draw.save(Path(draw_dir, "{}_ok.png".format(Path(annos[0]).name)))
            

        boxes = np.array(y[:,:4],dtype=np.float32)
        boxes[:,0] = boxes[:,0]/self.image_size[1]
        boxes[:,1] = boxes[:,1]/self.image_size[0]
        boxes[:,2] = boxes[:,2]/self.image_size[1]
        boxes[:,3] = boxes[:,3]/self.image_size[0]
        boxes = np.maximum(np.minimum(boxes,1),0)
        
        y = np.concatenate([boxes, y[:,-1:]],axis=-1)
        return img, y

    def __getitem__(self, index):
        if self.mosaic:
            img, y = self.load_index_img(index)
            imgs = [img]
            ys = [y]
            for i in range(3):
                img_y = self.load_index_img(np.random.randint(0, len(self)))
                imgs.append(img_y[0])
                ys.append(img_y[1])
            img, y = get_mosaic(imgs, ys, input_shape=self.image_size[0:2])
        else:
            img, y = self.load_index_img(index)
            
        img = np.array(img, dtype = np.float32)
        tmp_inp = np.transpose(img - MEANS, (2,0,1))
        tmp_targets = np.array(y, dtype = np.float32)

        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        if box is None:
            continue
        images.append(img)
        bboxes.append(box)
    if len(bboxes)==0:
        return None, None
    images = np.array(images)
    return images, bboxes



def get_mosaic(imgs, labels, input_shape):
    # labels: list of xmin,ymin,xmax,ymax,y_index 坐标归一化过
    h, w = input_shape
    min_offset_x = 0.3
    min_offset_y = 0.3
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
    for image, box in zip(imgs, labels):
        image = Image.fromarray(image.astype(np.uint8))

        # 图片的大小
        iw, ih = image.size

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h),
                                (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))
    # 归一化
    new_boxes[:,[0,2]] = new_boxes[:,[0,2]]/w
    new_boxes[:,[1,3]] = new_boxes[:,[1,3]]/h

    if len(new_boxes) == 0:
        return new_image, []
    if (new_boxes[:, :4] > 0).any():
        return new_image, new_boxes
    else:
        return new_image, []

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue
                
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue
                
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2-y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2-x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a    