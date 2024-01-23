import zipfile
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from dc_model_repo import model_repo_client
from dc_model_repo.pipeline.pipeline import DCPipeline

import get_map
import get_performance


def get_pipeline(work_dir, model_uri, is_debug_model):
    # 读入模型
    model_tmp_path = work_dir + "/model.zip"
    model_path = work_dir + "/model"
    model_repo_client.get(model_uri, model_tmp_path, timeout=(2, 60))

    # 解压模型文件
    def unzip_file(zip_src, dst_dir):
        r = zipfile.is_zipfile(zip_src)
        if r:
            fz = zipfile.ZipFile(zip_src, 'r')
            for file in fz.namelist():
                fz.extract(file, dst_dir)
        else:
            print('This is not zip')

    unzip_file(model_tmp_path, model_path)

    # 加载模型
    print("model_path:", model_path)
    pipeline = DCPipeline.load(model_path, 'local', debug_log=is_debug_model)
    pipeline.prepare(confidence=0.01)

    return pipeline


def check_img(img_path):
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


def to_df_voc(cur_ds):
    def _get_label_from_xml(xml_file):

        try:
            p = Path(xml_file)
            root = ET.parse(p).getroot()
            target_data = []
            for obj in root.iter('object'):
                cls = obj.find('name').text

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


def to_df_coco(cur_coco):
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


def get_df(cur_ds):
    valid_ds(cur_ds)
    if cur_ds.label_format == "VOC":
        df_all = to_df_voc(cur_ds)
    elif cur_ds.label_format == "COCO":
        df_all = to_df_coco(cur_ds.data)
    else:
        raise Exception('不支持这种数据类型:{}'.format(cur_ds.label_format))
    print("样本条数：", len(df_all["target"]))
    return df_all
    
    
def evaluate(predictions, work_dir):
    mAP, class_list = get_map.calculate_map(predictions, work_dir)
    performance = get_performance.calculate_performance(work_dir, class_list)
    
    for i in performance:
        if i['name'] in ["precision_recall_curve", "precision_curve", "recall_curve", "f1_curve"]:
            for j in i['data'].values():
                steps = int(len(j['charData'])/500)
                if steps == 0:
                    steps = 1
                j['charData'] = j['charData'][::steps]
    
    return performance