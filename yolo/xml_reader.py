import os
import xml.etree.cElementTree as cET

# 用于将xml文件中的图片名字以及坐标获取到，并转换为中心点坐标和宽高的形式存到txt文档中
xml_path = r"C:\Users\liev\Pictures\Camera Roll\outputs"
save_file = r"C:\Users\liev\Pictures\Camera Roll\result\xml.txt"
class_name = {
    1: "牛",
    2: "马",
    3: "人",
    4: "椅子",
    5: "猫",
    6: "狗",
    7: "猴",
    8: "石头",
    9: "飞机",
    0: "车"
}


# 根据值找到字典的键，前提是键值对都是唯一的，返回的是一个列表,返回列表中的第一个元素
def get_key(d, value):
    k = [k for k, v in d.items() if v == value]
    return k[0]


try:
    xml_txt = open(save_file, 'w')
    for file in os.listdir(xml_path):
        # print(file)
        file = os.path.join(xml_path, file)
        tree = cET.parse(file)
        root = tree.getroot()
        pic_name = root.findtext('filename')
        # print(pic_name)
        xml_txt.write(pic_name)
        for j in root.iter('object'):
            tag_name = j.findtext('name')
            # print(get_key(class_name, tag_name))
            tag_num = get_key(class_name, tag_name)
            for i in j.iter('bndbox'):
                x_min = int(i.findtext('xmin'))
                y_min = int(i.findtext('ymin'))
                x_max = int(i.findtext('xmax'))
                y_max = int(i.findtext('ymax'))
                cx = (x_max + x_min) // 2
                cy = (y_max + y_min) // 2
                w = x_max - x_min
                h = y_max - y_min
                print(cx, cy, w, h)
                xml_txt.write(" " + str(tag_num) + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h))
        xml_txt.write("\n")
except Exception as e:
    print("文件操作异常", e)
finally:
    xml_txt.close()
