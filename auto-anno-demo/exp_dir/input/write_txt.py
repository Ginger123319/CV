import pandas as pd
import json

df = pd.read_csv("input_label.csv")
from collections import Counter

cnt = Counter()
for a in df["label"]:
    anno = json.loads(a)
    cnt.update([v["category_id"] for v in anno["annotations"]])

cat_dict = {v: i for i, v in enumerate(sorted(cnt.keys()))}
print(cat_dict)
wh = (100, 200)

from pathlib import Path

img_dir = "image"
label_dir = "label"
Path(img_dir).mkdir(exist_ok=True)
Path(label_dir).mkdir(exist_ok=True)


def convert_box(wh, box):
    xmin, ymin, xmax, ymax = [float(v) for v in box]
    dw, dh = 1. / wh[0], 1. / wh[1]
    x, y, w, h = (xmin+xmax) / 2.0 - 1, (ymin+ymax) / 2.0 - 1, xmax-xmin, ymax-ymin
    return [x * dw, y * dh, w * dw, h * dh]


def get_wh(img_path):
    import cv2
    img = cv2.imread(img_path)
    return img.shape[1], img.shape[0]


for i, row in df.iterrows():
    s_id, s_img, s_label = row["sampleId"], row["path"], row["label"]
    img_name = f"{s_id}{Path(s_img).suffix}"
    label_name = f"{s_id}.txt"
    with open(Path(label_dir, label_name), 'w') as out_file:
        for anno in json.loads(s_label)["annotations"]:
            class_id = str(cat_dict[anno["category_id"]])
            center_xywh = convert_box(wh=get_wh(s_img), box=anno["bbox"])
            out_file.write("{} {} {} {} {}".format(class_id, *center_xywh) + "\n")