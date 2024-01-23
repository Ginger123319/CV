import os
import cv2
import xml.etree.cElementTree as ET

# 统一的思路就是打开再去保存这些特殊格式的文件，保存的时候可以进行重命名
# # 图片文件批量重命名
# root = r"G:\liewei\source\YOLO\song_yolov3data\img_song"
# new_root = "VOC2007/JPEGImages"
# for i, elem in enumerate(os.listdir(root)):
#     print(elem)
#     i += 11
#     if os.path.isfile(os.path.join(root, elem)):
#         new_name = str(i) + ".jpeg"
#         print(new_name)
#         print(elem)
#         new_img = cv2.imread(os.path.join(root, elem))
#         cv2.imwrite(os.path.join(new_root, new_name), new_img)

# xml文件批量重命名
xml_root = r"G:\liewei\source\YOLO\song_yolov3data\outputs"
new_path = "VOC2007/Annotations"
for i, elem in enumerate(os.listdir(xml_root)):
    print(elem)
    i += 11
    new_name = str(i) + ".xml"
    print(new_name)
    if os.path.isfile(os.path.join(xml_root, elem)):
        doc = ET.parse(os.path.join(xml_root, elem))
        doc.write(os.path.join(new_path, new_name), encoding='utf-8')
