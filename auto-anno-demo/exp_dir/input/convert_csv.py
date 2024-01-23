# 将目标检测数据集转成csv
img_dir = "/home/zk/code/dataset/VOC2007/image/"
xml_dir = "/home/zk/code/dataset/VOC2007/label/"
# img_dir = "/home/k/Documents/datasets/voc/extract/VOCdevkit/VOC2007_test/JPEGImages"
# xml_dir = "/home/k/Documents/datasets/voc/extract/VOCdevkit/VOC2007_test/Annotations"

from pathlib import Path

imgs = {p.stem: p for p in Path(img_dir).iterdir() if p.is_file() and p.suffix.lower() == ".jpg"}
xmls = {p.stem: p for p in Path(xml_dir).iterdir() if p.is_file() and p.suffix.lower() == ".xml"}

samples = sorted(list(set(imgs.keys()).intersection(xmls.keys())))

import pandas as pd


def extract_xml(xml_path):
    import xml.etree.ElementTree as ET
    from pathlib import Path
    if isinstance(xml_path, (str, Path)):
        element = ET.parse(xml_path).getroot()
    elif isinstance(xml_path, ET.ElementTree):
        element = xml_path.getroot()
    elif isinstance(xml_path, ET.Element):
        element = xml_path
    else:
        raise Exception(f"Unsupported xml input: {xml_path}")

    assert isinstance(element, ET.Element)
    content = {}
    for child in element:
        tag = child.tag
        child_content = extract_xml(child)[tag]
        if tag not in content:
            content[tag] = child_content
        elif tag in content:
            if isinstance(content[tag], list):
                content[tag].append(child_content)
            else:
                content[tag] = [content[tag]]
    if len(content) == 0:
        text = element.text
        if text is None or text.strip() == "":
            content = ""
        else:
            content = text.strip()
    return {element.tag: content}


col_id = []
col_path = []
col_label = []

for v in samples:
    col_id.append(v)
    col_path.append(str(imgs[v]))
    xml = extract_xml(xmls[v])
    objects = xml["annotation"]["object"]
    anno = []
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        anno.append({"category_id": obj["name"],
                     "bbox": [obj["bndbox"]["xmin"],
                              obj["bndbox"]["ymin"],
                              obj["bndbox"]["xmax"],
                              obj["bndbox"]["ymax"]]})
    import json
    col_label.append(json.dumps({"annotations": anno}))

df = pd.DataFrame({"sampleId":col_id, "path":col_path, "label":col_label})
from sklearn.model_selection import train_test_split

df_label, df_raw = train_test_split(df, test_size=0.1, random_state=6, shuffle=True)
df_init, df_anno = train_test_split(df_label, test_size=0.1, random_state=6, shuffle=True)
df_init.to_csv("input_label.csv", index=False)
print("Init cnt:", df_init.shape[0])
df_anno.to_csv("input_added_label.csv", index=False)
print("Add cnt:", df_anno.shape[0])
df_raw.drop(columns=["label"], inplace=True)
print("Raw cnt:", df_raw.shape[0])
df_raw.to_csv("input_unlabel.csv", index=False)


