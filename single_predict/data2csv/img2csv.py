import os
import csv
import pandas as pd
import xml.etree.ElementTree as ET

# 按目录分类数据集转换
from pycocotools.coco import COCO


def dir2csv(img_dir, csv_name, csv_column, is_train=True):
    if os.path.exists(csv_name):
        os.remove(csv_name)
    class_names = os.listdir(img_dir)
    # 遍历所有分类数据集中的图片
    data = []
    id_counter = 0
    for class_name in class_names:
        class_dir = os.path.join(img_dir, class_name)
        images = os.listdir(class_dir)
        for img in images:
            id_counter += 1
            img_path = os.path.join(class_dir, img)
            if is_train:
                label = {'annotations': [{'category_id': class_name}]}
            else:
                label = None
            data.append({"id": id_counter, "path": img_path, "label": label})

    # 将数据写入csv文件
    with open(csv_name, mode="w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_column)
        writer.writeheader()
        for item in data:
            writer.writerow(item)
    print("dir2csv convert done!!!!!")


# voc格式，一个文件夹放图片，一个文件夹放xml文件
def voc2csv(xml_path, img_path, csv_name, csv_column, is_train=True):
    num = 0
    if os.path.exists(csv_name):
        os.remove(csv_name)

    for img_file in os.listdir(img_path):
        num += 1
        if num % 5 == 0:
            index = img_file.rfind(".")
            suffix = img_file[index + 1:]
            xml_file = os.path.join(xml_path, img_file.replace(suffix, "xml"))
            img_file = os.path.join(img_path, img_file)

            if os.path.exists(xml_file):
                label_root = ET.parse(xml_file).getroot()
                label_dict = {"annotations": []}
                for obj in label_root.iter('object'):
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        continue

                    cls = obj.find('name').text
                    if is_train:
                        label_dict["annotations"].append({"category_id": cls})
                    else:
                        label_dict = None
                    # 如果xml中有多个类别，只取第一个
                    break

                df = pd.DataFrame(data=[[str(num), str(img_file), label_dict]])
                if not os.path.exists(csv_name):
                    df.to_csv(csv_name, header=csv_column, index=False, mode='a')
                else:
                    df.to_csv(csv_name, header=False, index=False, mode='a')
            else:
                print("Skip--xml not found:{}".format(img_file))
                continue


print("voc2csv convert done!!!!!")


# coco格式，一个文件夹放图片，在这个文件夹的同级目录存放标注文件annotation.json
def coco2csv(data_dir, anno_file, csv_name, csv_column, is_train=True):
    if os.path.exists(csv_name):
        os.remove(csv_name)
    # 加载标注文件
    coco = COCO(anno_file)
    # 遍历所有图片，并将图片路径和标签存储到一个列表中
    data = []
    id_counter = 0
    for i in coco.getImgIds():
        id_counter += 1
        img = coco.imgs[i]
        img_file = img["file_name"]
        img_path = os.path.join(data_dir, img_file)
        annos = coco.imgToAnns.get(i, None)
        label_dict = {"annotations": []}
        tmp_label = ''
        for anno in annos:
            tmp_label = coco.loadCats(anno['category_id'])[0]["name"]
        if tmp_label == '':
            continue
        if is_train:
            label_dict["annotations"].append({"category_id": tmp_label})
        else:
            label_dict = None
        data.append({"id": id_counter, "path": img_path, "label": label_dict})

    # 将数据写入csv文件
    with open(csv_name, mode="w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_column)
        writer.writeheader()
        for item in data:
            writer.writerow(item)
    print("coco2csv convert done!!!!!")


if __name__ == '__main__':
    # ----------------------------------------------------------
    # 1.修改参数is_train来修改转换模式
    #   训练模式，生成csv中有标签；测试模式，生成的csv中没有标签
    # 2.修改data_type指定数据集的类型
    # 3.修改对应的路径为自定义路径即可
    # ----------------------------------------------------------
    # 定义csv文件的路径和列名
    is_train = True
    if is_train:
        csv_file = r"../exp_dir/input/input_label.csv"
    else:
        csv_file = r"../exp_dir/input/input_unlabeled.csv"

    dir_name = os.path.dirname(csv_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    csv_columns = ["id", "path", "label"]

    # data_type ['dir','coco','voc']
    data_type = 'voc'
    if data_type == 'dir':
        # 按目录分类数据集
        data_dir = r"D:/Python/test_jzyj/images/三分类"
        dir2csv(data_dir, csv_file, csv_columns, is_train=is_train)
    elif data_type == 'voc':
        # voc格式数据集
        xml_path = r"D:/download/DatasetId_1791208_1679636382/DatasetId_1791208_1679636382/Annotations"
        img_path = r"D:/download/DatasetId_1791208_1679636382/DatasetId_1791208_1679636382/Images"
        voc2csv(xml_path, img_path, csv_file, csv_columns, is_train=is_train)
    else:
        # coco格式数据集
        img_path = r'D:/download/coco_classification_sigle_label_standard_folder/image'
        anno_path = r'D:/download/coco_classification_sigle_label_standard_folder/annotation.json'
        coco2csv(img_path, anno_path, csv_file, csv_columns, is_train=is_train)
