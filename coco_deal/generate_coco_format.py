import json
import os

import cv2

root = r'glue_tubes_keypoints_dataset_134imgs/train'
imgs_files = sorted(os.listdir(os.path.join(root, "images")))
annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
print(len(imgs_files) == len(annotations_files))

coco_dict = {
    'images': [],
    'annotations': [],
    'categories': []
}
coco_dict['categories'].append(
    {
        "supercategory": "glue",
        "id": 1,
        "name": "glue"
    }
)

ann_idx = 0
for idx in range(len(imgs_files)):
    img_path = os.path.join(root, "images", imgs_files[idx])
    img_original = cv2.imread(img_path)
    # cv2.imshow('a',img_original)
    # cv2.waitKey(0)
    h, w, c = img_original.shape
    coco_dict['images'].append({
        "license": 3,
        "file_name": imgs_files[idx],
        "height": h,
        "width": w,
        "id": idx
    })
    annotations_path = os.path.join(root, "annotations", annotations_files[idx])
    with open(annotations_path) as f:
        data = json.load(f)
        bboxes_original = data['bboxes']
        keypoints_original = data['keypoints']
        # print(bboxes_original,keypoints_original)
        for i, bbox in enumerate(bboxes_original):
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            for keypoint in keypoints_original[i]:
                if keypoint[-1] == 1:
                    keypoint[-1] = 2
            keypoints = keypoints_original[i][0] + keypoints_original[i][1]

            coco_dict['annotations'].append({
                "segmentation": [[]],
                "num_keypoints": len(keypoints) // 3,
                "area": bbox[-1]*bbox[-2],
                "iscrowd": 0,
                "keypoints": keypoints,
                "image_id": idx,
                "bbox": bbox,
                "category_id": 1,
                "id": ann_idx
            })
            ann_idx += 1
# 保存拆分后的数据集的标注信息
with open(root + '/annotation.json', 'w') as f:
    json.dump(coco_dict, f)
