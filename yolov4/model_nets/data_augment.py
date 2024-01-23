import os
import string

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from model_nets.utils.utils import merge_bboxes

MEANS = (104, 117, 123)


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def augment_train_sample_with_mosica(annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
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
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image_path = line_content[0]
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

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

        # 进行色域变换
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
        image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

        image = Image.fromarray((image * 255).astype(np.uint8))
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
        return image_path, new_image, []
    if (new_boxes[:, :4] > 0).any():
        return image_path, new_image, new_boxes
    else:
        return image_path, new_image, []


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
        return image_path, image_data, np.array([])

    if (box_data[:, :4] > 0).any():
        return image_path, image_data, box_data
    else:
        return image_path, image_data, np.array([])


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


def augment_train_data(annotation_data: list, tf_record_file: string, input_shape: tuple):
    def example_generator():
        for row in annotation_data:
            image_path, image_data, label = augment_train_sample(row, input_shape=input_shape)
            # if label.shape[0]==0:
            #     print(f"TFRecord skip {image_path}: no label.")
            #     continue
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


def augment_val_data(annotation_data: list, tf_record_file: string, input_shape: tuple):
    def example_generator():
        for row in annotation_data:
            image_path, image_data, label = augment_val_sample(row, input_shape=input_shape)
            # if label.shape[0]==0:
            #     print(f"TFRecord skip {image_path}: no label.")
            #     continue
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
