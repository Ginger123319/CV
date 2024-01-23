import os
import xml.etree.cElementTree as cET

# 用于将xml文件中的图片名字以及坐标获取到，并转换为中心点坐标和宽高的形式存到txt文档中
# xml_path = r"C:\Users\liev\Pictures\Camera Roll\outputs"
# song
xml_path = r"D:\Python\test_jzyj\augment\labels"
save_file = r"data\label_ori.txt"
class_name = {
    1: "键盘",
    0: "鼠标"
}

color_name = {
    1: "red",
    0: "black"
}


# 根据值找到字典的键，前提是键值对都是唯一的，返回的是一个列表,返回列表中的第一个元素
def get_key(d, value):
    k = [k for k, v in d.items() if v == value]
    return k[0]


if __name__ == '__main__':
    xml_txt = open(save_file, 'w')
    try:
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
                    # xml_txt.write(" " + str(tag_num) + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h))
                    xml_txt.write(
                        " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + " " + str(tag_num))
            xml_txt.write("\n")
    except Exception as e:
        print("文件操作异常", e)
    finally:
        xml_txt.close()
