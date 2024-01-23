from datacanvas.aps import dc
import sys
import os
import glob
from xml.etree import ElementTree as ET
import pandas as pd
import re
import shutil
import json
from pathlib import Path

# 获取图片数据集中的ground_truth信息
def get_gt(df, classes, work_dir):
    assert isinstance(df, pd.DataFrame)
    for i, row in df.iterrows():
        img_path = row["path"]
        target = row["target"]
        #if "'" in raw_target:
        #    raw_target = raw_target.replace("'", '"')
        #target = json.loads(raw_target)
        image_id = Path(img_path).stem
        target_filtered = [d for d in target if d["label"] in classes]
        if len(target_filtered)>0:
            f_path = Path(work_dir, "middle_dir_mask_dis/image_info/ground_truth/", "{}.txt".format(image_id))
            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)
            new_f = open(f_path, "w")
            for t in target_filtered:
                new_f.write("{} {} {} {} {}\n".format(t["label"], t["left"], t["top"], t["right"], t["bottom"]))
            new_f.close()
    print("ground truth conversion completed!")
