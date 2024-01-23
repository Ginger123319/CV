import os
import string

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MEANS = (104, 117, 123)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def augment_train_sample(annotation_data: string, input_shape: tuple, jitter=.3, hue=.1, sat=1.5, val=1.5):
    '''r实时数据增强的随机预处理'''
    line = annotation_data.split()
    image_path = line[0]
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.5, 1.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
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
        return image_path, image_data, []

    if (box_data[:, :4] > 0).any():
        return image_path, image_data, box_data
    else:
        return image_path, image_data, []


def augment_val_sample(annotation_data: string, input_shape: tuple, jitter=.3, hue=.1, sat=1.5, val=1.5):
    line = annotation_data.split()
    image_path = line[0]
    image = Image.open(image_path)
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)

    # correct boxes
    box_data = np.zeros((len(box), 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box

    return image_path, image_data, box_data


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # if isinstance(value, type(tf.constant(0))):
    #    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tf_record(tf_record_file, example_generator):
    if os.path.exists(tf_record_file):
        os.remove(tf_record_file)

    writer = tf.io.TFRecordWriter(tf_record_file)
    try:
        for _, tf_example in enumerate(example_generator):
            writer.write(tf_example.SerializeToString())
    except Exception as e:
        raise e
    finally:
        writer.close()


def augment_train_data(annotation_data: list, tf_record_file: string):
    def example_generator():
        for row in annotation_data:
            image_path, image_data, label = augment_train_sample(row, input_shape=(416, 416))
            if label.shape[0]==0:
                print(f"TFRecord skip {image_path}: no label.")
                continue
            feature = {
                "path": _bytes_feature(str(image_path).encode(encoding="utf-8")),
                "image": _bytes_feature(image_data.tobytes()),
                "image_shape": _bytes_feature(
                    str(",".join([str(i) for i in image_data.shape])).encode(encoding="utf-8")),
                "label": _bytes_feature(label.tobytes()),
                "label_shape": _bytes_feature(str(",".join([str(i) for i in label.shape])).encode(encoding="utf-8"))
            }
            yield tf.train.Example(features=tf.train.Features(feature=feature))

    write_tf_record(tf_record_file=tf_record_file, example_generator=example_generator())


def augment_val_data(annotation_data: list, tf_record_file: string):
    def example_generator():
        for row in annotation_data:
            image_path, image_data, label = augment_val_sample(row, input_shape=(416, 416))
            if label.shape[0]==0:
                print(f"TFRecord skip {image_path}: no label.")
                continue
            feature = {
                "path": _bytes_feature(str(image_path).encode(encoding="utf-8")),
                "image": _bytes_feature(image_data.tobytes()),
                "image_shape": _bytes_feature(
                    str(",".join([str(i) for i in image_data.shape])).encode(encoding="utf-8")),
                "label": _bytes_feature(label.tobytes()),
                "label_shape": _bytes_feature(str(",".join([str(i) for i in label.shape])).encode(encoding="utf-8"))
            }
            yield tf.train.Example(features=tf.train.Features(feature=feature))

    write_tf_record(tf_record_file=tf_record_file, example_generator=example_generator())
