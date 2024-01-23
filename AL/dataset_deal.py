import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class COCODataset(object):
    # dataset是一个COCO对象
    def __init__(self, transforms, dataset, class_name, dataset_len):
        self.transforms = transforms
        self.coco = dataset
        self.class_name = class_name
        self.dataset_len = dataset_len
        self.ids = list(self.coco.imgToAnns.keys())

        self.class_map = dict()
        for k, v in self.coco.cats.items():
            self.class_map[k] = v['name']

    def parse_label(self, target):
        labels = []
        boxes = []
        for obj in target:
            bbox = obj['bbox']
            boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])
            labels.append(self.class_name.index(self.class_map[obj['category_id']]))
        return labels, boxes

    def __getitem__(self, index):
        image_id = self.ids[index]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        img_path = self.coco.loadImgs(image_id)[0]["file_full_path"]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        masks = np.array([self.coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
        masks = masks.reshape((-1, height, width))
        labels, boxes = self.parse_label(target)

        num_objs = len(masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.dataset_len


class InsSegDataset(object):
    # 此处dataset是一个Dataset对象
    def __init__(self, transforms, dataset, class_name, dataset_len):
        self.transforms = transforms
        self.dataset = dataset
        self.class_name = class_name
        self.dataset_len = dataset_len

    def parse_label(self, xml_path):
        label_root = ET.parse(xml_path).getroot()

        labels = []
        boxes = []
        for obj in label_root.iter('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue

            cls = obj.find('name').text
            labels.append(self.class_name.index(cls))

            xmin = int(obj.find("bndbox").find('xmin').text)
            ymin = int(obj.find("bndbox").find('ymin').text)
            xmax = int(obj.find("bndbox").find('xmax').text)
            ymax = int(obj.find("bndbox").find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return labels, boxes

    def __getitem__(self, index):
        img_path = self.dataset.get_item(index).data
        mask_path = self.dataset.get_item(index).mask
        label_path = self.dataset.get_item(index).label

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        labels, boxes = self.parse_label(label_path)

        # 删除mask中的白色部分
        mask[mask == 255] = 0
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    from pycocotools.coco import COCO

    json_file = r"D:\download\data_20221206103845\json同目录\annotation.json"
    coco = COCO(json_file)
    coco_dataset = COCODataset(dataset=coco)
