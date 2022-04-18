import numpy as np
import os
from xml.dom.minidom import parse
import xml.dom.minidom
from PIL import Image, UnidentifiedImageError

# bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 1, 18, 17, 13]])
# print(bs)
# # 降序排序
# flag = bs[:, -1]
# print(flag)
#
# bs = bs[np.argsort(-flag)]
# print(bs)
#
# test = bs[:, 2]
# index = np.where(test < 13)
# print(type(index))
# print(test)

# 打开两个文件夹，遍历所有文件
pic_path = r"D:\markHelper\data\solo_pic"
xml_path = r"D:\markHelper\data\solo_pic\outputs"
result_path = r"D:\markHelper\data\solo_pic\result"

f = open(os.path.join(result_path, "solo_pic.txt"), mode="w")
f.write("solo_pic\n")
f.write("image_id " + "x_1 " + "y_1 " + "width " + "height\n")
counter = 0
# 先遍历xml，筛选标记过的xml，尝试打开图片，打不开就跳过，打得开再判断是否符合要求
for xml_name in os.listdir(xml_path):
    DOMTree = xml.dom.minidom.parse(os.path.join(xml_path, xml_name))
    collection = DOMTree.documentElement
    labeled = collection.getElementsByTagName('labeled')
    # 通过labeled标签之间的数据来判断当前图片是否被标记过
    # labeled[0]表示第一个labeled标签
    # print(labeled[0].firstChild.data)
    if labeled[0].firstChild.data == "false":
        # print("未标记{}".format(xml_name))
        continue
    path = collection.getElementsByTagName('path')[0].firstChild.data
    pic_dir = os.path.join(pic_path, path.split("\\")[-1])

    try:
        img = Image.open(pic_dir)
    except UnidentifiedImageError:
        continue
    else:
        shape = np.shape(img)
        if len(shape) == 3 and shape[0] > 100 and shape[1] > 100 and shape[2] == 3:
            img = img
        else:
            continue
        # print(pic_dir)
        # print(xml_name)
        # print(collection.getElementsByTagName('depth')[0].firstChild.data)
        image_id = path.split("\\")[-1]

        x_1 = int(collection.getElementsByTagName('xmin')[0].firstChild.data)
        y_1 = int(collection.getElementsByTagName('ymin')[0].firstChild.data)
        x_2 = int(collection.getElementsByTagName('xmax')[0].firstChild.data)
        y_2 = int(collection.getElementsByTagName('ymax')[0].firstChild.data)
        width = x_2 - x_1

        height = y_2 - y_1
        f.write(image_id + " " + str(x_1) + " " + str(y_1) + " " + str(width) + " " + str(height) + "\n")
        counter += 1
        # print(width, height)
    # if i == 3:
    #     break
f.close()
print("写入数据{}".format(counter))
# 不符合要求就跳过，符合要求再去处理xml，将需要的字段写到txt文件中
