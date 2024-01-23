def check_img(img_path):
    from pathlib import Path
    p = Path(img_path)
    if not p.is_file():
        print("文件不存在: {}".format(img_path))
        return False
    if p.stat().st_size <= 0:
        print("文件大小不能为0: {}".format(img_path))
        return False
    return True


def valid_ds(cur_ds):
    assert cur_ds.data_type == "image", "数据集类型不对"
    assert cur_ds.label_type == "object_detection", "数据集类型不对"
    assert cur_ds.label_format in ["VOC", "COCO"], "数据集类型不对"


# [{"left": 9, "top": 61, "right": 218, "bottom": 282, "label": "horse"}, {"left": 301, "top": 76, "right": 500,
# "bottom": 282, "label": "horse"}]
def to_df_voc(cur_ds, classes):
    import pandas as pd

    def _get_label_from_xml(xml_file):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        try:
            p = Path(xml_file)
            root = ET.parse(p).getroot()
            target_data = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                box = obj.find('bndbox')
                b = {'left': int(box.find('xmin').text), 'top': int(box.find('ymin').text),
                     'right': int(box.find('xmax').text),
                     'bottom': int(box.find('ymax').text), 'label': cls}
                target_data.append(b)
            if len(target_data) == 0:
                print("SKIP: 没有找到标签：{}".format(xml_file))
                return None
            else:
                return target_data
        except Exception as e:
            print("SKIP：解析文件失败：{}".format(p.resolve()))
            return None

    content = {"path": [], "target": []}
    for img in cur_ds.data:
        if check_img(img.data):
            tmp_label = _get_label_from_xml(xml_file=img.label)
            if tmp_label is not None:
                content["path"].append(img.data)
                content["target"].append(tmp_label)
    df = pd.DataFrame(content)
    return df


def to_df_coco(cur_coco, classes):
    import pandas as pd
    content = {"path": [], "target": []}
    for img_id in cur_coco.getImgIds():
        img = cur_coco.imgs[img_id]
        # img_path = img["file_name"]
        img_path = img["file_full_path"]
        if check_img(img_path):
            annos = cur_coco.imgToAnns.get(img_id, None)
            if annos is None:
                print("跳过,没有标注：{}".format(img_path))
                continue
            target_data = []
            for anno in annos:
                bbox = anno.get("bbox", None)
                cat = anno.get("category_id", None)
                if bbox and cat:
                    cat_str = cur_coco.loadCats(cat)[0]["name"]
                    if cat_str not in classes:
                        continue
                    x, y, w, h = bbox
                    target_data.append(
                        {"left": int(x), "top": int(y), "right": int(x + w), "bottom": int(y + h), "label": cat_str})
            if len(target_data) == 0:
                print("没有找到标签：{}".format(img_path))
                continue
            content["path"].append(img_path)
            content["target"].append(target_data)
    df = pd.DataFrame(content)
    return df


def get_df(cur_ds, test_size, seed, classes):
    from sklearn.model_selection import train_test_split

    valid_ds(cur_ds)

    if cur_ds.label_format == "VOC":
        df_all = to_df_voc(cur_ds, classes=classes)
    elif cur_ds.label_format == "COCO":
        df_all = to_df_coco(cur_ds.data, classes=classes)
    else:
        raise Exception

    print("样本条数：", len(df_all["target"]))
    df_train, df_valid = train_test_split(df_all, test_size=test_size, random_state=seed, shuffle=True)
    return df_train, df_valid
