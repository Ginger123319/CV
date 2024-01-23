import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from model_nets.utils.utils import merge_bboxes
import cv2

import io
import os
import struct
# from tkinter.messagebox import NO
import typing
import tfrecord
from torch.utils.data.dataset import Dataset

class YoloDataset2(Dataset):
    def __init__(self, train_lines, tf_record_file, index_file, image_size, mosaic=True):
        super(YoloDataset2, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.mosaic = mosaic
        self.flag = True
        self.tf_record_file = tf_record_file
        self.index_file = index_file
        self.indexs = np.loadtxt(index_file, dtype=np.int64)
        self.file_io = None
        
    def __len__(self):
        return self.train_batches

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

    def __getitem__(self, index):
        record_iterator = self._tfrecord_iterator(self.tf_record_file, self.indexs, idx=index)
        for record in record_iterator:
            feature = self._parse_record(record)        
        
        img_shape = [int(i) for i in bytes(feature['image_shape']).decode("utf-8").split(",")]
        y_shape = [int(i) for i in bytes(feature['label_shape']).decode("utf-8").split(",")]
        
        img = np.frombuffer(feature['image'], dtype=np.float32).reshape(img_shape)
        y = np.frombuffer(feature['label'], dtype=float).reshape(y_shape)
        if y.shape[0]==0:
            return None, None
        
        
        # lines = self.train_lines
        # n = self.train_batches
        # index = index % n
        # if self.mosaic:
        #     if self.flag and (index + 4) < n:
        #         img, y = self.get_random_data_with_Mosaic(lines[index:index + 4], self.image_size[0:2])
        #     else:
        #         img, y = self.get_random_data(lines[index], self.image_size[0:2])
        #     self.flag = bool(1-self.flag)
        # else:
        #     img, y = self.get_random_data(lines[index], self.image_size[0:2])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        if tmp_targets.shape[0]==0:
            return None, None
        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

