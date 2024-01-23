import io
import os
import struct
# from tkinter.messagebox import NO
import typing
import numpy as np
import tfrecord
from torch.utils.data.dataset import Dataset
from PIL import Image

MEANS = (104, 117, 123)


class SSDDataset2(Dataset):
    def __init__(self, train_samples, tf_record_file, index_file, image_size: tuple, mosaic=False):
        super(SSDDataset2, self).__init__()

        self.mosaic = mosaic
        self.train_samples = train_samples
        self.sample_num = len(train_samples)
        self.tf_record_file = tf_record_file
        self.index_file = index_file
        self.indexs = np.loadtxt(index_file, dtype=np.int64)
        # 统一一张验证集和多张验证集数据的indexs格式
        if len(self.indexs.shape) == 1:
            self.indexs = np.array([self.indexs])
        self.image_size = image_size
        self.file_io = None

        self.total_time = 0.0

    def __len__(self):
        return self.sample_num

    def _tfrecord_iterator(self, data_path: str, index: np.array, idx: int) -> typing.Iterable[memoryview]:
        if self.file_io is None:
            self.file_io = io.open(data_path, "rb")
        
        length_bytes = bytearray(8)
        crc_bytes = bytearray(4)
        datum_bytes = bytearray(1024 * 1024)

        def read_records(start_offset=None, end_offset=None):
            nonlocal length_bytes, crc_bytes, datum_bytes

            if start_offset is not None:
                self.file_io.seek(start_offset)
            if end_offset is None:
                end_offset = os.path.getsize(data_path)
            while self.file_io.tell() < end_offset:
                if self.file_io.readinto(length_bytes) != 8:
                    raise RuntimeError("Failed to read the record size.")
                if self.file_io.readinto(crc_bytes) != 4:
                    raise RuntimeError("Failed to read the start token.")
                length, = struct.unpack_from("<Q", length_bytes)
                if length > len(datum_bytes):
                    datum_bytes = datum_bytes.zfill(int(length * 1.5))
                datum_bytes_view = memoryview(datum_bytes)[:length]
                if self.file_io.readinto(datum_bytes_view) != length:
                    raise RuntimeError("Failed to read the record.")
                if self.file_io.readinto(crc_bytes) != 4:
                    raise RuntimeError("Failed to read the end token.")
                yield datum_bytes_view

        # start_offset = index[idx]
        # end_offset = index[idx + 1] if idx < (len(index) - 1) else None

        start_offset = index[idx][0]
        end_offset = start_offset + index[idx][1]
        yield from read_records(start_offset, end_offset)

    def _parse_record(self, record, description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None):
        example = tfrecord.example_pb2.Example()
        example.ParseFromString(record)

        all_keys = list(example.features.feature.keys())
        if description is None:
            description = dict.fromkeys(all_keys, None)
        elif isinstance(description, list):
            description = dict.fromkeys(description, None)

        features = {}
        for key, typename in description.items():
            if key not in all_keys:
                raise KeyError(
                    f"Key {key} doesn't exist (select from {all_keys})!")
            # NOTE: We assume that each key in the example has only one field
            # (either "bytes_list", "float_list", or "int64_list")!
            field = example.features.feature[key].ListFields()[0]
            inferred_typename, value = field[0].name, field[1].value
            if typename is not None:
                tf_typename = self.typename_mapping[typename]
                if tf_typename != inferred_typename:
                    reversed_mapping = {v: k for k,
                                        v in self.typename_mapping.items()}
                    raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                                    f"(should be '{reversed_mapping[inferred_typename]}').")

            # Decode raw bytes into respective data types
            if inferred_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif inferred_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif inferred_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value
        return features

    def load_index_img(self, index):
        record_iterator = self._tfrecord_iterator(
            self.tf_record_file, self.indexs, idx=index)
        for record in record_iterator:
            feature = self._parse_record(record)
        
        img_shape = [int(i) for i in bytes(feature['image_shape']).decode("utf-8").split(",")]
        y_shape = [int(i) for i in bytes(feature['label_shape']).decode("utf-8").split(",")]
        img = np.frombuffer(feature['image'], dtype=np.float32).reshape(img_shape)
        y = np.frombuffer(feature['label'], dtype=float).reshape(y_shape)

        boxes = np.array(y[:, :4], dtype=np.float32)
        boxes[:, 0] = boxes[:, 0]/self.image_size[1]
        boxes[:, 1] = boxes[:, 1]/self.image_size[0]
        boxes[:, 2] = boxes[:, 2]/self.image_size[1]
        boxes[:, 3] = boxes[:, 3]/self.image_size[0]
        boxes = np.maximum(np.minimum(boxes, 1), 0)

        y = np.concatenate([boxes, y[:, -1:]], axis=-1)
        return img, y

    def __getitem__(self, index):
        img, y = self.load_index_img(index)
        if y.shape[0]==0:
            return None, None
        if self.mosaic:
            imgs = [img]
            ys = [y]
            for i in range(3):
                img_y = self.load_index_img(np.random.randint(0, len(self)))
                if len(img_y[1])==0:
                    return None, None
                imgs.append(img_y[0])
                ys.append(img_y[1])
            img, y = get_mosaic(imgs, ys, input_shape=self.image_size[0:2])
            if len(y)==0:
                return None, None

        img = np.array(img, dtype=np.float32)
        tmp_inp = np.transpose(img - MEANS, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        if tmp_targets.shape[0]==0:
            return None, None

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
    if len(new_boxes) == 0:
        return new_image, []
    # 归一化
    new_boxes[:,[0,2]] = new_boxes[:,[0,2]]/w
    new_boxes[:,[1,3]] = new_boxes[:,[1,3]]/h

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
