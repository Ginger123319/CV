from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


class FRCNNDataset(Dataset):
    def __init__(self, train_lines, shape=[600, 600], batch_size=1, mosaic=False):
        self.mosaic = mosaic
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.shape = shape
        self.batch_size = batch_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = self.shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def load_index_img(self, index):
        lines = self.train_lines
        n = self.train_batches
        while True:
            img, y = self.get_random_data(lines[index])
            if len(y) == 0:
                index = (index + 1) % n
                continue
            box = np.array(y[:, :4], dtype=np.float32)
            box[:, 0] = y[:, 0]
            box[:, 1] = y[:, 1]
            box[:, 2] = y[:, 2]
            box[:, 3] = y[:, 3]

            box_widths = box[:, 2] - box[:, 0]
            box_heights = box[:, 3] - box[:, 1]
            if (box_heights <= 0).any() or (box_widths <= 0).any():
                index = (index + 1) % n
                continue
            break

        label = y[:, -1]
        return img, box, label
        
    def __getitem__(self, index):
        img, box, label = self.load_index_img(index)
        if self.mosaic:
            imgs = [img]
            y = np.concatenate([box, label.reshape(-1,1)], axis=1)
            y[:,[1,3]] = y[:,[1,3]]/img.shape[0]
            y[:,[0,2]] = y[:,[0,2]]/img.shape[1]
            ys = [y]
            for i in range(3):
                img_y = self.load_index_img(np.random.randint(0, len(self)))
                imgs.append(img_y[0])
                y = np.concatenate([img_y[1], img_y[2].reshape(-1,1)], axis=1)
                
                ys.append(y)
            img, y = get_mosaic(imgs, ys, input_shape=self.shape)
            
            #print("mosaic-y:\n", y)
            if len(y)==0:
                #print("Mosaic出来的样本无label，跳过：", index)
                return self.__getitem__(np.random.randint(0, len(self)))
            y[:,[1,3]] = y[:,[1,3]]*img.shape[0]
            y[:,[0,2]] = y[:,[0,2]]*img.shape[1]
            box = y[:,:4]
            label = y[:,4]
            
        img = np.transpose(img / 255.0, [2, 0, 1])
        return img, box, label


# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    return images, bboxes, labels


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

    if len(new_boxes) == 0:
        return new_image, []
    if (new_boxes[:, :4] > 0).any():
        # 归一化
        new_boxes[:,[0,2]] = new_boxes[:,[0,2]]/w
        new_boxes[:,[1,3]] = new_boxes[:,[1,3]]/h
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