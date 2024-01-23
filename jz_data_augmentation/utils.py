import pandas as pd
import ast
import sys
import numpy as np
from PIL import Image

# 校验图片路径以及图片大小
cnt = 0
cnt1 = 0


def check_img(img_path):
    global cnt1
    cnt1 += 1
    from pathlib import Path
    p = Path(img_path)
    if not p.exists():
        global cnt
        cnt += 1
        print(cnt, "/", cnt1, "文件不存在: {}".format(img_path))
        return False
    if p.stat().st_size <= 0:
        print("文件大小不能为0: {}".format(img_path))
        return False
    return True


# 校验dataset
def valid_ds(cur_ds):
    assert cur_ds.data_type == "image", "数据集类型不对"
    assert cur_ds.label_type in ["object_detection", "image_classification",
                                 "image_classification_multi_label"], "数据集类型不对"
    assert cur_ds.label_format in ["VOC", "COCO", "imagefolder"], "数据集类型不对"


# 读取图片
def read_rgb_img(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != "RGB" and img.mode != "RGBA":
            print("Don't support this image mode: {}. Skip it: {}".format(img.mode, img_path))
            return "Image error: mode {}".format(img.mode)
        img = img.convert("RGB")
        if img.width < 1 or img.height < 1:
            print(
                "This image has strange height [{}] or width [{}]. Skip it: {}".format(img.height, img.width, img_path))
            return "Image error: size {}x{}".format(img.height, img.width)
        return np.array(img)
    except Exception as e:
        print("Error [{}] while reading image. Skip it: {}".format(e, img_path))
        return "Image error: {}".format(str(e))


# 图像分类-单标签-按目录
def to_df_imagefolder(cur_ds, labels_selected_list, image_col="image_path", label_col="label"):
    content = {image_col: [], label_col: []}
    for img in cur_ds.data:
        if check_img(img.data):
            if img.label is not None:
                label = [img.label]
                # 标签筛选
                if len(labels_selected_list) != 0 and img.label not in labels_selected_list:
                    print("图片标签与选择的标签不匹配,不做数据增强! {}:{}".format(img.label, img.data))
                    continue
            else:
                label = []
            content[image_col].append(img.data)
            content[label_col].append(label)
    df = pd.DataFrame(content)
    return df


# voc
def to_df_voc(cur_ds, label_type, labels_selected_list, image_col="image_path", label_col="label"):
    def _get_label_from_xml(xml_file):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        try:
            p = Path(xml_file)
            root = ET.parse(p).getroot()
            height = int(root.find('size').find('height').text)
            width = int(root.find('size').find('width').text)
            label = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if label_type == "103":
                    xmin = int(obj.find("bndbox").find('xmin').text)
                    ymin = int(obj.find("bndbox").find('ymin').text)
                    xmax = int(obj.find("bndbox").find('xmax').text)
                    ymax = int(obj.find("bndbox").find('ymax').text)
                    # 过滤框的边界并进行修改
                    x = np.array([xmin, xmax])
                    y = np.array([ymin, ymax])
                    x = np.clip(x, 0, width)
                    y = np.clip(y, 0, height)
                    label.append([x[0], y[0], x[1], y[1], cls])
                else:
                    label.append(cls)
            return label
        except Exception as e:
            print("SKIP：解析文件失败：{}".format(p.resolve()))
            print("error is {}".format(e))
            return None

    content = {image_col: [], label_col: []}
    for img in cur_ds.data:
        if check_img(img.data):
            if img.label is not None:
                tmp_label = _get_label_from_xml(img.label)
            else:
                tmp_label = []
            # 标签筛选
            if tmp_label is not None and len(tmp_label) != 0:
                if label_type == "103":
                    label_list = [label[-1] for label in tmp_label]
                else:
                    label_list = tmp_label

                if len(labels_selected_list) != 0 and len(set(label_list) & set(labels_selected_list)) == 0:
                    print("图片标签与选择的标签不匹配,不做数据增强! {}:{}".format(label_list, img.data))
                    continue
            if tmp_label is not None:
                content[image_col].append(img.data)
                content[label_col].append(tmp_label)
    df = pd.DataFrame(content)
    return df


# coco
def to_df_coco(cur_coco, label_type, labels_selected_list, image_col="image_path", label_col="label"):
    # 某个图片如果有多个标注，不用这个图片
    content = {image_col: [], label_col: []}

    for img_id in cur_coco.getImgIds():

        img = cur_coco.imgs[img_id]
        img_path = img["file_full_path"]
        height = img['height']
        width = img['width']

        if check_img(img_path):
            annos = cur_coco.imgToAnns.get(img_id, None)
            tmp_label = []
            if annos is not None:
                for anno in annos:
                    cat = anno.get("category_id", None)
                    if cat:
                        cat_str = cur_coco.loadCats(cat)[0]["name"]
                        if label_type == "103":
                            bbox = anno.get("bbox", None)
                            if bbox:
                                x, y, w, h = bbox
                                xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)

                                x = np.array([xmin, xmax])
                                y = np.array([ymin, ymax])
                                x = np.clip(x, 0, width)
                                y = np.clip(y, 0, height)
                                tmp_label.append([x[0], y[0], x[1], y[1], cat_str])
                        else:
                            tmp_label.append(cat_str)

            # 标签筛选
            if len(tmp_label) != 0:
                if label_type == "103":
                    label_list = [label[-1] for label in tmp_label]
                else:
                    label_list = tmp_label
                if len(labels_selected_list) != 0 and len(set(label_list) & set(labels_selected_list)) == 0:
                    print("图片标签与选择的标签不匹配,不做数据增强! {}:{}".format(label_list, img_path))
                    continue
            content[image_col].append(img_path)
            content[label_col].append(tmp_label)
    df = pd.DataFrame(content)
    return df


def get_image_list(cur_ds, image_col, label_col, label_type, labels_selected_list, is_augment_no_labeled_data):
    valid_ds(cur_ds)

    if cur_ds.label_format == "imagefolder":
        df_all = to_df_imagefolder(cur_ds, labels_selected_list, image_col=image_col, label_col=label_col)
    elif cur_ds.label_format == "VOC":
        df_all = to_df_voc(cur_ds, label_type, labels_selected_list, image_col=image_col, label_col=label_col)
    elif cur_ds.label_format == "COCO":
        df_all = to_df_coco(cur_ds.data, label_type, labels_selected_list, image_col=image_col, label_col=label_col)
    else:
        raise Exception

    df = df_all

    # 是否对未标注数据进行增强
    if not is_augment_no_labeled_data and not df.empty:
        df = df[~df[label_col].apply(lambda x: [] == x)]
        print("未标注数据不增强！")

    image_list = []
    class_name = {}
    file_name = []
    for img_path, labels in zip(df[image_col], df[label_col]):
        img = read_rgb_img(img_path)
        if isinstance(img, str):
            continue
        file_name.append(img_path)

        label_list = []
        for label in labels:
            if label_type == "103":
                cls = label[-1]
            else:
                cls = label
            if cls not in class_name.keys():
                class_name[cls] = len(class_name) + 1
            if label_type == "103":
                label_list.append(label[:4] + [class_name[cls]])
            else:
                label_list.append(class_name[cls])
        image_list.append([img, label_list])

    return image_list, class_name, file_name


def get_methods(augment_methods):
    methods_list = []
    if augment_methods == "":
        sys.exit("用户未选择任何增强方法！")
    temp_list = ast.literal_eval(augment_methods)
    if len(temp_list) == 0:
        sys.exit("用户未选择任何增强方法！")
    else:
        methods_list = [method['name'] for method in temp_list]

    return methods_list


def get_labels_selected(labels_selected):
    labels_list = []

    if labels_selected == "":
        print("用户未进行标签筛选！")
    else:
        temp_list = ast.literal_eval(labels_selected)
        labels_list = [label['label'] for label in temp_list]

    return labels_list


def get_image_list_mix_mosaic(image, label, image_list, label_type):
    tmp_list1 = []
    tmp_list2 = []
    for image_label_mix_mosaic in image_list:
        # 排除当前图片，避免自身和自身做混合操作
        if image.all != image_label_mix_mosaic[0].all and len(image_label_mix_mosaic[1]) != 0:
            tmp_list1.append([image_label_mix_mosaic[0], image_label_mix_mosaic[1]])
            # 单分类时，将同类别的图片放在一个目录中
            if label_type == "101" and label == image_label_mix_mosaic[1]:
                tmp_list2.append([image_label_mix_mosaic[0], image_label_mix_mosaic[1]])
    if label_type == '101':
        image_list_mix_mosaic = tmp_list2
    else:
        image_list_mix_mosaic = tmp_list1
    return image_list_mix_mosaic


def create_annotation(annotation, relative_path, filename, img, labels, class_name, label_type):
    h, w = img.shape[:2]
    if annotation['images'] == []:
        image_id = 0
    else:
        image_id = annotation['images'][-1]['id'] + 1
    annotation['images'].append({
        "license": 0,
        "file_url": relative_path,
        "file_name": filename,
        "height": h,
        "width": w,
        "date_captured": "",
        "id": image_id
    })

    if len(labels) != 0:
        if label_type == "103":
            for label in labels:

                if not annotation['annotations']:
                    anno_id = 0
                else:
                    anno_id = annotation['annotations'][-1]['id'] + 1
                xmin, ymin, xmax, ymax, cls = label
                bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
                annotation['annotations'].append({
                    'id': anno_id,
                    'image_id': image_id,
                    'category_id': int(cls),
                    'segmentation': [],
                    'area': 0,
                    'bbox': bbox,
                    'iscrowd': 0
                })
        else:
            for _, label in enumerate(labels):
                if not annotation['annotations']:
                    anno_id = 0
                else:
                    anno_id = annotation['annotations'][-1]['id'] + 1
                annotation['annotations'].append({
                    'id': anno_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'segmentation': [],
                    'area': 0,
                    'bbox': [],
                    'iscrowd': 0
                })
    return annotation
