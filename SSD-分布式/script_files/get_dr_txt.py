import colorsys
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from model_nets.nets.ssd import get_ssd
from model_nets.ssd2 import SSD
from model_nets.utils.box_utils import letterbox_image, ssd_correct_boxes
import re
from datacanvas.aps import dc

MEANS = (104, 117, 123)


class mAP_SSD(SSD):
    def generate(self):
        self.confidence = 0.01
        # -------------------------------#
        #   计算总的类的数量
        # -------------------------------#
        self.num_classes = len(self.class_names) + 1

        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        model = get_ssd("test", self.num_classes, self.confidence, self.nms_iou)
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image, detect_result_path):

        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1], self.input_shape[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)

        photo = np.array(crop_img, dtype=np.float64)
        with torch.no_grad():
            photo = Variable(
                torch.from_numpy(np.expand_dims(np.transpose(photo - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)

            top_conf = []
            top_label = []
            top_bboxes = []
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        if len(top_conf) <= 0:
            return

        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
        # -----------------------------------------------------------#
        #   去掉灰条部分
        # -----------------------------------------------------------#
        if self.letterbox_image:
            boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                      np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        f = open(detect_result_path + "/" + image_id + ".txt", "w")
        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    # 检测图片，获取图片数据集中的detect_result信息


def get_dr(df, image_size, work_dir):
    from pathlib import Path
    model_dir_list = os.listdir(work_dir + '/middle_dir_ssd_dis/normal_train_best_model_dir')

    for model_index in model_dir_list:
        # 遍历model_dir中的pth权重文件，分别创建detect_result目录
        if model_index.endswith('.pth'):
            detect_result_path = work_dir + "/middle_dir_ssd_dis/image_info/detect_result_" + model_index
            os.makedirs(detect_result_path)

            ssd = mAP_SSD(image_size, work_dir)
            for i in range(len(df)):
                p = Path(df["path"][i])
                image_id = p.stem
                image_path = str(p.absolute())
                image = Image.open(image_path)
                image = image.convert("RGB")
                ssd.detect_image(image_id, image, detect_result_path)

    print("detect result conversion completed!")
