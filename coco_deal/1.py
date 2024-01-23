import json
import os
import random
import cv2
from PIL import Image

random.seed(1)

image_path = r"D:\download\archive\images"
json_file = r"D:\download\archive\keypoints.json"

pic_nums = []
coco_dict = {
    'images': [],
    'annotations': [],
    'categories': []
}
with open(json_file) as f:
    data = json.load(f)
    for i in range(1, len(data['images']) + 1, 11):
        img = cv2.imread(os.path.join(image_path, data["images"][str(i)]))
        h, w, c = img.shape
        # cv2.imshow('pic',img)
        # cv2.waitKey()
        # exit()
        coco_dict['images'].append({
            "license": 3,
            "file_name": data["images"][str(i)],
            "height": h,
            "width": w,
            "id": i
        })
        pic_nums.append(i)
    anno_id = 0
    for anno in data['annotations']:
        if anno['image_id'] in pic_nums:
            bbox_original = anno['bbox']
            keypoints_original = anno['keypoints']
            bbox = [bbox_original[0], bbox_original[1], bbox_original[2] - bbox_original[0],
                    bbox_original[3] - bbox_original[1]]

            for keypoints in keypoints_original:
                if keypoints[-1] == 1:
                    keypoints[-1] = 2
            keypoints = sum(keypoints_original, [])
            coco_dict['annotations'].append({
                "num_keypoints": anno['num_keypoints'],
                "area": bbox[-1] * bbox[-2],
                "iscrowd": 0,
                "keypoints": keypoints,
                "image_id": anno['image_id'],
                "bbox": bbox,
                "category_id": 1,
                "id": anno_id
            })
            anno_id += 1
    coco_dict['categories'] = data['categories']
# 保存拆分后的数据集的标注信息
with open("D:/download/archive" + '/annotation.json', 'w') as f:
    json.dump(coco_dict, f)
