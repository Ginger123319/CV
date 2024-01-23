import os
import re 
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model_nets.frcnn2 import FRCNN
from model_nets.nets.frcnn import FasterRCNN
from model_nets.nets.frcnn_training import get_new_img_size
from model_nets.utils.utils import DecodeBox
from datacanvas.aps import dc 


class mAP_FRCNN(FRCNN):
    
    #   检测图片
    def detect_image(self, image_id, image, detect_result_path):
        self.confidence = 0.01
        self.iou = 0.45
        f = open(detect_result_path+ "/" + image_id + ".txt", "w")
        
        with torch.no_grad():
            image_shape = np.array(np.shape(image)[0:2])
            old_width = image_shape[1]
            old_height = image_shape[0]
            width, height = get_new_img_size(old_width, old_height)

            image = image.resize([width, height], Image.BICUBIC)
            photo = np.array(image, dtype=np.float32) / 255
            photo = np.transpose(photo, (2, 0, 1))

            images = []
            images.append(photo)
            images = np.asarray(images)
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, roi_indices = self.model(images)
            decodebox = DecodeBox(self.std, self.mean, self.num_classes)
            outputs = decodebox.forward(roi_cls_locs, roi_scores, rois, height=height, width=width, nms_iou=self.iou,
                                        score_thresh=self.confidence)
            if len(outputs) == 0:
                return
            bbox = outputs[:, :4]
            conf = outputs[:, 4]
            label = outputs[:, 5]

            bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
            bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
            bbox = np.array(bbox, np.int32)

        for i, c in enumerate(label):
            predicted_class = self.class_names[int(c)]
            score = str(conf[i])

            left, top, right, bottom = bbox[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

# 检测图片，获取图片数据集中的detect_result信息
def get_dr(df, image_size, work_dir):
    from pathlib import Path
    
    model_dir_list = os.listdir(work_dir+'/middle_dir_mask_dis/normal_train_best_model_dir')
    
    for model_index in model_dir_list:
        # 遍历model_dir中的pth权重文件，分别创建detect_result目录
        if model_index.endswith('.pth'):
            detect_result_path = work_dir+"/middle_dir_mask_dis/image_info/detect_result_" + model_index
            os.makedirs(detect_result_path)

            frcnn = mAP_FRCNN(work_dir)
            for i in range(len(df)):
                p = Path(df["path"][i])
                image_id = p.stem
                image_path = str(p.absolute())
                image = Image.open(image_path)
                image = image.convert("RGB")
                frcnn.detect_image(image_id, image, detect_result_path)                   
    print("detect result conversion completed!")


