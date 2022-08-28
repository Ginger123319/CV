import csv
import ast
import os
import time

from PIL import ImageDraw, Image

csv_path = r"D:\Python\source\fire\csv\190910_fire_4802张-数据回传.csv"
img_path = r"D:\Python\source\fire\fire_1000"
label_path = r"D:\Python\source\fire\labels\train"


# 将左上角右下角坐标转换成中心点和宽高偏移量的形式
# 归一化方式才采取除以对应的宽和高的方式
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    name_list = []
    count = 0
    for i, row in enumerate(reader):
        # print("points--{}".format(row["region_shape_attributes"]))
        # print(type(row["region_shape_attributes"]))
        point_dict = ast.literal_eval(row["region_shape_attributes"])
        # print(type(point_dict))
        # print(type(point_dict["all_points_x"]))
        x_list = point_dict["all_points_x"]
        y_list = point_dict["all_points_y"]
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)
        # print(x_min, y_min, x_max, y_max)
        img_name = row["#filename"]
        name_list.append(img_name)
        if img_name in os.listdir(img_path):
            # print(img_name, name_list[i - 1])
            if i == 0 or (i > 0 and img_name != name_list[i - 1]):
                count += 1
                label_file = open(os.path.join(label_path, img_name).replace("jpg", "txt"), mode='w')
                image = Image.open(os.path.join(img_path, img_name))
                size = image.size
                draw = ImageDraw.Draw(image)
            # draw.rectangle([x_min, y_min, x_max, y_max], outline='blue', width=2)
            # image.show()
            # time.sleep(3)
            # 将类别和坐标写到标签对应的txt中
            box = [x_min, x_max, y_min, y_max]
            bb = convert(size, box)
            label_file.write("0" + " " + " ".join([str(a) for a in bb]) + "\n")
    print(count)
