from datacanvas.aps import dc
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
import re
import colorsys
import torch.nn as nn

from model_nets.nets.yolo4 import YoloBody
from model_nets.utils.utils import DecodeBox, non_max_suppression, letterbox_image, yolo_correct_boxes

class YOLO(object):

    def __init__(self, model_path, image_size, work_dir):
        
        self.model_path = model_path
        self.model_image_size = (image_size, image_size, 3)
        self.anchors_path = 'script_files/yolo_anchors.txt'
        self.classes_path = work_dir+'/middle_dir/data_info/classes.txt'
        
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
    
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    def generate(self):
        self.net = YoloBody(len(self.anchors[0]),len(self.class_names)).eval()
        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('Finished!')
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
                
class mAP_Yolo(YOLO):
    
    def __init__(self, model_path, image_size, work_dir):
        super(mAP_Yolo, self).__init__(model_path, image_size, work_dir)
        
    # 检测图片
    def detect_image(self, image_id, image, model_index, detect_result_path):
        self.confidence = 0.01
        self.iou = 0.5
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence, nms_thres=self.iou)

        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return "no object has been detected!"
            
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin = np.expand_dims(top_bboxes[:,0],-1)
        top_ymin = np.expand_dims(top_bboxes[:,1],-1)
        top_xmax = np.expand_dims(top_bboxes[:,2],-1)
        top_ymax = np.expand_dims(top_bboxes[:,3],-1)
        
        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
        f = open(detect_result_path + '/' + image_id + ".txt", "w")
        
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])
            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

# 检测图片，获取图片数据集中的detect_result信息
def get_dr(df, image_size, work_dir):
    from pathlib import Path
    
    model_dir_list = os.listdir(work_dir+'/middle_dir/normal_train_best_model_dir')
    
    for model_index in model_dir_list:
        # 遍历model_dir中的pth权重文件，分别创建detect_result目录
        if model_index.endswith('.pth'):
            model_path =  work_dir+'/middle_dir/normal_train_best_model_dir/' + model_index
            detect_result_path = work_dir+"/middle_dir/image_info/detect_result_" + model_index
            os.makedirs(detect_result_path)

            yolo = mAP_Yolo(model_path, image_size, work_dir)
            for i in range(len(df)):
                p = Path(df["path"][i])
                image_id = p.stem
                image_path = str(p.absolute())
                image = Image.open(image_path)
                image = image.convert("RGB")
                yolo.detect_image(image_id, image, model_index, detect_result_path)
                
    print("detect result conversion completed!")


