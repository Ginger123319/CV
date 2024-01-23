cnt = 0
cnt1 = 0
def check_img(img_path):
    global cnt1
    cnt1+=1
    from pathlib import Path
    p = Path(img_path)
    if not p.exists():
        global cnt
        cnt+=1
        print(cnt, "/", cnt1, "文件不存在: {}".format(img_path))
        return False
    if p.stat().st_size <= 0:
        print("文件大小不能为0: {}".format(img_path))
        return False
    return True


def valid_ds(cur_ds):
    assert cur_ds.data_type == "image", "数据集类型不对"
    assert cur_ds.label_type == "image_classification", "数据集类型不对"
    assert cur_ds.label_format != "unlabeled", "数据集类型不对"


def to_df_imagefolder(cur_ds, image_col="image_path", label_col="label"):
    import pandas as pd

    content = {image_col: [], label_col: []}
    for img in cur_ds.data:
        if check_img(img.data):
            content[image_col].append(img.data)
            content[label_col].append(img.label)
    df = pd.DataFrame(content)
    return df


def to_df_voc(cur_ds, image_col="image_path", label_col="label"):
    # 如果有多个object，取第一个
    import pandas as pd

    def _get_label_from_xml(xml_file, label_path="./object/name"):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        try:
            p = Path(xml_file)
            root = ET.parse(p).getroot()
            label_ele = root.find(label_path)
            if label_ele is None:
                print("SKIP：在标注文件中没有找到标签：[{}] --> [{}]".format(p.resolve(), label_path))
                return None
            label = label_ele.text
            label = "" if label is None else label_ele.text.strip()
            if label == "":
                print("SKIP：在标注文件中标签为空：[{}] --> [{}]".format(p.resolve(), label_path))
                return None
            else:
                return label
        except Exception as e:
            print("SKIP：解析文件失败：{}".format(p.resolve()))
            return None

    content = {image_col: [], label_col: []}
    for img in cur_ds.data:
        if check_img(img.data):
            tmp_label = _get_label_from_xml(xml_file=img.label)
            if tmp_label is not None:
                content[image_col].append(img.data)
                content[label_col].append(tmp_label)
    df = pd.DataFrame(content)
    return df


def to_df_coco(cur_coco, image_col="image_path", label_col="label"):
    # 某个图片如果有多个标注，不用这个图片
    import pandas as pd
    content = {image_col: [], label_col: []}
    for img_id in cur_coco.getImgIds():
        img = cur_coco.imgs[img_id]
        # img_path = img["file_name"]
        img_path = img["file_full_path"]
        if check_img(img_path):
            annos = cur_coco.imgToAnns.get(img_id, None)
            if annos is None:
                print("跳过,没有标注：{}".format(img_path))
                continue
            if len(annos) != 1:
                print("跳过，当前图片标注不唯一：{}".format(img_path))
            tmp_label = cur_coco.loadCats(annos[0]['category_id'])[0]["name"]
            content[image_col].append(img_path)
            content[label_col].append(tmp_label)
    df = pd.DataFrame(content)
    return df


def get_df(cur_ds, test_size, seed=1, image_col="image_path", label_col="label"):
    from sklearn.model_selection import train_test_split

    valid_ds(cur_ds)

    if cur_ds.label_format == "imagefolder":
        df_all = to_df_imagefolder(cur_ds, image_col=image_col, label_col=label_col)
    elif cur_ds.label_format == "VOC":
        df_all = to_df_voc(cur_ds, image_col=image_col, label_col=label_col)
    elif cur_ds.label_format == "COCO":
        df_all = to_df_coco(cur_ds.data, image_col=image_col, label_col=label_col)
    else:
        raise Exception

    print("样本条数：", len(df_all["label"]))
    from collections import Counter
    c = Counter(df_all["label"])
    if len(c)<2:
        raise Exception("分类的类别至少是2，现在为：{}".format(len(c)))
    break_flag=False
    for k, v in c.items():
        print("类别[{}]个数：{}".format(k, v))
        if v<2:
            break_flag = True
    if break_flag:
        raise Exception("每个类别的样本数至少为2！")
    df_train, df_valid = train_test_split(df_all, test_size=test_size, random_state=seed, shuffle=True, stratify=df_all["label"])
    return df_train, df_valid