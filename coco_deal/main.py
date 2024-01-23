import os
import sys
import random
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2


# visualize func
def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 2, (255, 0, 0), 2)
            # image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:

        cv2.imshow('img', image)
        cv2.waitKey(0)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255, 0, 0), 10)
                # image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp),
                #                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)


# 划分数据集和标注文件
img_root = r"D:\download\archive\images"
json_file = r"D:\download\archive\annotation.json"
# 初始化 COCO 对象
coco = COCO(json_file)
# ---------------------------- 大json拆分
# 获取指定 image_id 对应的所有注释 ID
imgIds = coco.getImgIds()
# 随机打乱数据集
random.shuffle(imgIds)
# 计算训练集和验证集的数量
num_train = int(len(imgIds) * 0.9)
num_val = len(imgIds) - num_train
list_train = imgIds[:num_train]
list_val = imgIds[num_train:]
anns = coco.loadAnns(coco.getAnnIds(imgIds))
# for ann in anns:
#     if isinstance(ann['segmentation'], dict):
#         ann['segmentation'] = [ann['segmentation']['counts']]
imgs = coco.loadImgs(imgIds)
# print(imgs[0])
# exit(0)
cats = coco.loadCats(coco.getCatIds())
# 拷贝图片到对应的目录
for i, img in enumerate(imgs):
    img_path = os.path.join(img_root, img['file_name'])
    if i < num_train:
        shutil.copy(img_path, 'train2017')
        # pass
    else:
        shutil.copy(img_path, 'val2017')
# 拆分数据集
train_data = {
    'images': imgs[:num_train],
    'annotations': [ann for ann in anns if ann['image_id'] in list_train],
    'categories': cats
}

val_data = {
    'images': imgs[num_train:],
    'annotations': [ann for ann in anns if ann['image_id'] in list_val],
    'categories': cats
}

# 保存拆分后的数据集
with open('person_keypoints_train2017.json', 'w') as f:
    json.dump(train_data, f)

with open('person_keypoints_val2017.json', 'w') as f:
    json.dump(val_data, f)

# coco数据集常用api演示
# print(coco.dataset.keys())
# img_id_list = coco.getImgIds()
# print(img_id_list)
# catidlist = coco.getCatIds()
# print(catidlist)
# tmp_label = coco.loadCats(catidlist)
# print(tmp_label)
# #
# img_id = img_id_list[8]
# print(img_id)
# ann_ids = coco.getAnnIds(imgIds=[img_id])
# print(ann_ids)
# #
# anns = coco.loadAnns(ids=ann_ids)
# print(coco.cats)
# print(anns)

# 校验标注是否准确，可以可视化验证框和关键点的位置
# if __name__ == '__main__':
#     json_file = r'person_keypoints_val2017.json'
#     coco = COCO(json_file)
#     print(coco.dataset.keys())
#     # 类别信息
#     catIds = coco.getCatIds()
#     catInfo = coco.loadCats(catIds)
#     print(f"catIds:{catIds}")
#     print(f"catcls:{catInfo}")
#     # 图像信息
#     imgIds = coco.getImgIds()
#     for i in imgIds:
#         img = coco.imgs[i]
#         path = img["file_name"]
#         annos = coco.imgToAnns.get(i, None)
#         for anno in annos:
#             tmp_label = coco.loadCats(anno['category_id'])[0]["name"]
#         print(tmp_label)
#
#         index = 10  # 随便选择一张图
#         imgInfo = coco.loadImgs(i)[0]
#         print(os.path.join('val2017', imgInfo['file_name']))
#         # print(f"imgIds:{imgIds}")
#         print(f"img:{imgInfo}")
#         # 标注信息
#         annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
#         annsInfo = coco.loadAnns(annIds)
#         print(f"annIds:{annIds}")
#         print(f"annsInfo:{annsInfo}")
#
#         i = plt.imread(os.path.join('val2017', imgInfo['file_name']))
#         print(i.shape)
#         # break
#         # new
#         target = coco.loadAnns(coco.getAnnIds(imgIds=imgInfo['id']))
#         # print([len(i) for i in target[0]['segmentation']])
#         # try:
#         #     masks = np.array([coco.annToMask(obj).reshape(-1) for obj in target], np.float32)
#         # except Exception as e:
#         #     # 此处错误没有特别的颜色；和寻常打印一致
#         #     # 修改file类型，此时打印为红色了
#         #     print(repr(e), file=sys.stderr)
#         # print(masks.shape)
#
#         # 显示图像
#         # i = io.imread(imgInfo['coco_url'])
#         i = cv2.imread(os.path.join('val2017', imgInfo['file_name']))
#         keypoints = []
#         bboxes = []
#         for annos in annsInfo:
#             keypoint = annos['keypoints']
#             keypoint = [keypoint[i: i + 2] for i in range(0, len(keypoint), 3)]
#             bbox = annos['bbox']
#             bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
#             keypoints.append(keypoint)
#             bboxes.append(bbox)
#
#         visualize(i, bboxes, keypoints)
