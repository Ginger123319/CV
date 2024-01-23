import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .detection.engine import train_one_epoch, evaluate
from .detection import utils
from .detection import transforms as T
from . import cv2_util

import cv2
import random
import time
import datetime

def random_color():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
 
    return (b,g,r)


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256

def PredictImg(img, model, device, image_name, class_name, confidence):
    from datacanvas.aps import dc 
    #img, _ = dataset_test[0] 
    # img = cv2.imread(image)
    result = img.copy()
    dst=img.copy()
    original_image = img.copy()
    mask_image = img.copy()
    mask_image[mask_image != 0] = 0 #
    img=toTensor(img)

    # put the model in evaluati
    # on mode

    prediction = model([img.to(device)])

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks=prediction[0]['masks']

    m_bOK=False;
    for idx in range(boxes.shape[0]):

        if scores[idx] >= confidence:

            m_bOK=True
            color=random_color()
            mask=masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(dst, contours, -1, color, -1)
            cv2.drawContours(mask_image, contours, -1, color, -1)
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = class_name.get(str(labels[idx].item()))
            cv2.rectangle(result,(x1,y1),(x2,y2),color,thickness=2)
            cv2.putText(result, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

            dst1=cv2.addWeighted(result,0.7,dst,0.3,0)

    if m_bOK:
        # cv2.imshow('result',dst1)
        cv2.imwrite(str(dc.conf.outputs.prediction)+'/prediction_image/predict_'+image_name, dst1)
        Image.fromarray(mask_image).save(str(dc.conf.outputs.prediction)+'/mask_image/mask_'+image_name)
        cv2.imwrite(str(dc.conf.outputs.prediction)+'/original_image/'+image_name, original_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    else:
        print("模型未能检测出任何实例，或检测出的实例未能达到confidence阈值.")
        cv2.imwrite(str(dc.conf.outputs.prediction)+'/original_image/'+image_name, original_image)
        
def predicting(img, model, device, image_name, class_name, confidence):
    class_name = {str(i):j for i,j in enumerate(class_name)}
    PredictImg(img, model, device, image_name, class_name, confidence)


