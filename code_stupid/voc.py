import os
import xml.etree.ElementTree as ET
import cv2
import shutil
from PIL import Image
import numpy as np
import pandas as pd

xml_path = r"D:\Python\source\VOCdevkit\voc2012-500\xml"
img_path = r"D:\Python\source\VOCdevkit\voc2012-500\img"
mask_path = r"D:\Python\source\VOCdevkit\voc2012-500\mask"
# os.mkdir(os.path.join(mask_path, "mask"))
# os.mkdir(os.path.join(img_path, "img"))
# os.mkdir(os.path.join(xml_path, "xml"))
# xml_path = r"D:\Python\test_jzyj\key_mouse_20\xml"
# xml_path = r"D:\download\data_20221201111346\xml_standard\xml"
num = 0
path1 = 'temp.csv'
path2 = 'empty.csv'
for xml_file in os.listdir(xml_path):
    num += 1
    img_file = os.path.join(img_path, xml_file.replace("xml", "jpg"))
    xml_file = os.path.join(xml_path, xml_file)
    print(xml_file)


    # mask = np.array(Image.open(xml_file))
    # # 删除mask中的白色部分
    # mask[mask == 255] = 0
    # # instances are encoded as different colors
    # obj_ids = np.unique(mask)
    # # first id is the background, so remove it
    # obj_ids = obj_ids[1:]
    # # print(obj_ids)
    # # print(obj_ids[:, None, None])
    # # print(mask == obj_ids[:, None, None])
    # # split the color-encoded mask into a set of binary masks
    # masks = mask == obj_ids[:, None, None]
    #
    # # print(masks.shape)
    # # (4, 281, 500)
    # # 有四个掩码图，也就是四个目标
    #
    # # get bounding box coordinates for each mask
    # num_objs = len(obj_ids)
    # # exit()

    # if os.path.exists(img_file) and os.path.join(xml_file):
    #     shutil.copy(xml_file, os.path.join(mask_path, "mask"))
    #     shutil.copy(img_file, os.path.join(img_path, "img"))
    #     shutil.copy(xml_file, os.path.join(xml_path, "xml"))
    # if num >= 500:
    #     break

    # xml_file = xml_file.replace("xml", "jpg")
    # print(xml_file)
    # num += 1
    # xml_file = os.path.join(img_path, xml_file)
    # img = cv2.imread(xml_file)
    # cv2.putText(img, text="text!!!", org=(0, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))
    # cv2.imshow('text', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # img = Image.open(xml_file)
    # # img.show()
    # exit()
    label_root = ET.parse(xml_file).getroot()
    print(label_root)
    labels = []
    boxes = []
    label_dict = {"annotations": []}
    for obj in label_root.iter('object'):
        difficult = obj.find('difficult').text
        if int(difficult) == 1:
            continue

        cls = obj.find('name').text
        print(cls)
        # labels.append(self.class_name.index(cls))

        # xmin = int(obj.find("bndbox").find('xmin').text)
        # ymin = int(obj.find("bndbox").find('ymin').text)
        # xmax = int(obj.find("bndbox").find('xmax').text)
        # ymax = int(obj.find("bndbox").find('ymax').text)
        # boxes.append([xmin, ymin, xmax, ymax])
        #
        # print(boxes)
        label_dict["annotations"].append({"category_id": cls})
        break
    print(label_dict)
    if num % 5 != 0:
        path = path2
        df = pd.DataFrame(data=[[str(num), str(img_file), '']])
    else:
        path = path1
        df = pd.DataFrame(data=[[str(num), str(img_file), label_dict]])
    if not os.path.exists(path):
        df.to_csv(path, header=['id', 'path', 'label'], index=False, mode='a')
    else:
        df.to_csv(path, header=False, index=False, mode='a')

print(num)
