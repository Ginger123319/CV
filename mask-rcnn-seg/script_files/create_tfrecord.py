import random
import json
import requests
import shutil
import tensorflow as tf
from pathlib import Path
import math
import os
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET
from datacanvas.aps import dc 

# Constant
TASK_CLASSIFY = 'TASK_CLASSIFY'
TASK_DETECT = 'TASK_DETECT'
TASK_SEMANTIC = 'TASK_SEMANTIC'
TASK_INSTANCE = 'TASK_INSTANCE'
TASK_INFO = {TASK_CLASSIFY: "图像分类",
             TASK_DETECT: "目标检测",
             TASK_SEMANTIC: "语义分割",
             TASK_INSTANCE: "实例分割"}
FORMAT_VOC = 'FORMAT_VOC'
FORMAT_SAME_DIR = 'FORMAT_SAME_DIR'
FORMAT_DIRS = 'FORMAT_DIRS'
FORMAT_INFO = {FORMAT_VOC: "VOC分目录格式",
               FORMAT_SAME_DIR: "同目录格式",
               FORMAT_DIRS: "目录作为类别名格式"}
                   
def copy_dataset_dir(dataset_source_path, dataset_target_path):
    if not os.path.exists(dataset_source_path):
        raise Exception(f"{dataset_source_path} no such file or directory")

    target_parent_path = Path(dataset_target_path).parent
    if not target_parent_path.exists():
        os.makedirs(target_parent_path)

    if os.path.exists(dataset_target_path):
        shutil.rmtree(dataset_target_path)

    shutil.copytree(dataset_source_path, dataset_target_path)
    dc.logger.info("dataset copy dir finished!")


def call_java_dataset_update_progress(datasetId, progress):
    data = {
        "datasetId": datasetId,
        "progress": progress
    }
    post_headers = {'Content-Type': 'application/json', "Accept": "*/*"}
    # response = requests.post("http://192.168.56.1:1999/aps/internal/pipes/dataset/progress/update",data = json.dumps(data), headers=post_headers)
    response = requests.post("{}/aps/internal/pipes/dataset/progress/update".format(pipes_url), data=json.dumps(data), headers=post_headers)
    resp_body = response.json()
    if resp_body.get('code') != 0:
        raise Exception(resp_body)
    dc.logger.info("dataset update progress response：{}".format(response.text))


def call_java_dataset_calldas_analysis(datasetId, userId, projectId):
    data = {
        "datasetId": datasetId
    }
    post_headers = {'Content-Type': 'application/json', "Accept": "*/*", "userId": userId, "projectId": projectId}
    # response = requests.post("http://192.168.56.1:1999/aps/internal/pipes/dataset/calldas/analysis",data = json.dumps(data), headers=post_headers)
    response = requests.post("{}/aps/internal/pipes/dataset/calldas/analysis".format(pipes_url), data=json.dumps(data), headers=post_headers)
    resp_body = response.json()
    if resp_body.get('code') != 0:
        raise Exception(resp_body)
    dc.logger.info("dataset calldas analysis response：{}".format(response.text))


def calc_count_per_file(sample_cnt, size_total, file_cnt_max, size_per_file):
    size_per_sample = size_total / sample_cnt
    size_per_file = max(size_per_file, size_per_sample * 128)
    file_cnt = math.ceil(size_total / size_per_file)
    file_cnt = min(file_cnt, file_cnt_max)
    dc.logger.info("要生成{}个tfrecord文件".format(file_cnt))
    cnt_per_file = math.ceil(sample_cnt / file_cnt)
    dc.logger.info("每个文件中放{}个样本".format(cnt_per_file))
    return cnt_per_file


def parse_label(xml_file):
    root = ET.parse(xml_file).getroot()
    label_list = []
    for obj in root.iter('object'):
        try:
            label_list.append(obj.find('name').text)
        except:
            print('this object is invalid, will be ignored!')
            continue
    return label_list


def write_tf_record(tf_record_dir, cnt_per_file, example_generator):
    if tf_record_dir.exists():
        for f in tf_record_dir.iterdir():
            if f.is_file() and f.suffix == ".tfrecord":
                dc.logger.info("删除旧的tfrecord文件：{}".format(f.resolve()))
                f.unlink()
    else:
        tf_record_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    file_index = -1
    cnt = 0
    try:
        for i, tf_example in enumerate(example_generator):
            if i % cnt_per_file == 0:
                if writer is not None:
                    writer.close()
                cnt = 0
                file_index += 1
                cur_tfrecord = Path(tf_record_dir, "{}_data_{}.tfrecord".format(file_index, cnt_per_file))
                cur_tfrecord_str = str(cur_tfrecord.resolve())
                dc.logger.info("创建tfrecord文件: {}".format(cur_tfrecord_str))
                writer = tf.io.TFRecordWriter(cur_tfrecord_str)
            cnt += 1
            writer.write(tf_example.SerializeToString())
        writer.close()
        if cnt != cnt_per_file:
            dst = Path(tf_record_dir, "{}_data_{}.tfrecord".format(file_index, cnt))
            if dst.exists():
                dc.logger.info("删除旧的tfrecord文件：{}".format(dst.resolve()))
                dst.unlink()
            dc.logger.info("重命名tfrecord文件: {} --> {}".format(cur_tfrecord.resolve(), dst.resolve()))
            os.rename(src=cur_tfrecord, dst=dst)
    except Exception as e:
        if writer is not None:
            writer.close()
        raise e


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # if isinstance(value, type(tf.constant(0))):
    #    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_label_from_xml(xml_file, label_path):
    try:
        p = xml_file
        root = ET.parse(p).getroot()
        label_ele = root.find(label_path)
        if label_ele is None:
            dc.logger.info("SKIP：在标注文件中没有找到标签：[{}] --> [{}]".format(p.resolve(), label_path))
            return None
        label = label_ele.text
        label = "" if label is None else label_ele.text.strip()
        if label == "":
            dc.logger.info("SKIP：在标注文件中标签为空：[{}] --> [{}]".format(p.resolve(), label_path))
            return None
        else:
            return label
    except Exception as e:
        dc.logger.info("SKIP：解析文件失败：{}".format(p.resolve()))
        return None


def get_files_from_dir(cur_dir, suffix=None, label_from_xml=None):
    dc.logger.info("提取目录中的文件信息: {}".format(cur_dir))
    assert isinstance(cur_dir, Path) and cur_dir.is_dir()
    file_info = dict()
    for p in cur_dir.iterdir():
        if p.is_file() and p.stat().st_size > 0 and (suffix is None or p.suffix == suffix) and (not p.name.startswith(".")):
            # 保证图像分类场景下可以从xml里获取标签
            if suffix == ".xml" and label_from_xml is not None:
                if _get_label_from_xml(p, label_from_xml) is None:
                    continue
            file_info[p.stem] = p
        else:
            dc.logger.info("SKIP: {}".format(p.resolve()))
    return file_info


# function
def convert_tfrecord(dataset_path,
                     task_type,
                     format_type,
                     data_detail,
                     tfrecord_num_max=20,
                     tfrecord_size=150 * 1024 ** 2,
                     tfrecord_output_path="tfrecord",
                     csv_output_path="info.csv"):
    """
    :dataset_path 数据集文件路径
    :task_type 任务类型，数字：
     - 1代表图像分类
     - 2代表目标检测
     - 3代表语义分割
     - 4代表实例分割
    :format_type 格式标识，数字：
     - 1代表VOC指定各类文件所在文件夹的格式
     - 2代表标住文件与图片在同一目录
     - 3代表图像分类以文件夹作为类别的数据
    :data_detail 数据集详情，Dict。下面描述中的dir都是相对dataset_path的相对路径。
     - format_type==1时：{"image":img_dir,"label":label_xml_dir,"mask":mask_dir}

       校验规则：只有task_type in [1,2,3,4] 时可用，且只有task_type in [3,4] 时才能有mask

     - format_type==2时：{"image_and_label": image_and_label_dir}

       校验规则：只有task_type in [1,2]时可用

     - format_type==3时，{"image_categories": parent_dir_of_images}

       校验规则：只有task_type==1时可用，其中parent_dir_of_images指存放图片文件的目录的父目录

    :tfrecord_num_max TFRECORD的拆分个数最大值
    :tfrecord_size TFRECORD单个文件的默认大小，字节为单位
    :tfrecord_output_path TFRECORD文件的输出目录，相对dataset_path的相对路径。
    :csv_output_path 要生成的CSV文件保存路径，相对dataset_path的相对路径。
    :return: None
    """
    start_time = datetime.now()
    dc.logger.info("CONVERT_TFRECORD 开始 ----- {}".format(start_time.isoformat()))
    dataset_loc = Path(dataset_path)
    assert dataset_loc.exists() and dataset_loc.is_dir()
    assert task_type in (TASK_CLASSIFY, TASK_DETECT, TASK_SEMANTIC, TASK_INSTANCE)
    assert format_type in (FORMAT_VOC, FORMAT_SAME_DIR, FORMAT_DIRS)
    assert len(data_detail) > 0

    dc.logger.info("任务：{}".format(TASK_INFO.get(task_type)))
    dc.logger.info("数据集格式： {}".format(FORMAT_INFO.get(format_type)))

    tfrecord_loc = Path(dataset_loc, tfrecord_output_path)

    def write_csv(df):
        csv_path = Path(dataset_loc, csv_output_path)
        if csv_path.exists() and csv_path.is_file():
            dc.logger.info("删除旧的CSV文件：{}".format(csv_path.resolve()))
            csv_path.unlink()
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True)
        dc.logger.info("写CSV文件:{}".format(csv_path))
        df.to_csv(csv_path, index=False)

    if format_type == FORMAT_VOC:
        assert task_type in (TASK_CLASSIFY, TASK_DETECT, TASK_SEMANTIC, TASK_INSTANCE), "VOC格式数据集只能用于图像分类、目标检测、语义分割、实例分割任务。 当前任务类型：{}".format(task_type)

        sample_info = dict()
        assert "image" in data_detail and "label" in data_detail

        sample_info["image"] = get_files_from_dir(Path(dataset_loc, data_detail["image"]))

        # 校验文件，必须有"./object/name"节点
        sample_info["label"] = get_files_from_dir(Path(dataset_loc, data_detail["label"]), suffix=".xml", label_from_xml="./object/name")

        if task_type in (TASK_SEMANTIC, TASK_INSTANCE):
            assert "mask" in data_detail
            sample_info["mask"] = get_files_from_dir(Path(dataset_loc, data_detail["mask"]))

        dc.logger.info("筛选有效样本...")
        names_valid = None
        for k, v in sample_info.items():
            if names_valid is None:
                names_valid = set(v)
            else:
                names_valid = names_valid.intersection(set(v))
        sample_cnt = len(names_valid)
        dc.logger.info("有效样本数量：{}，原始图片数量：{}".format(len(names_valid), len(sample_info["image"])))
        assert sample_cnt > 0

        shuffled_names = list(names_valid)
        random.shuffle(shuffled_names)

        dc.logger.info("整理要输出的csv数据...")
        csv_content = {"path": [], "label": []}
        for name in shuffled_names:
            csv_content["path"].append(str(sample_info["image"][name].resolve()))
            if task_type == TASK_CLASSIFY:
                csv_content["label"].append(_get_label_from_xml(sample_info["label"][name], "./object/name"))
            elif task_type in (TASK_DETECT, TASK_INSTANCE):
                csv_content["label"].append(";".join(parse_label(sample_info["label"][name])))
        df_write = pd.DataFrame(data=csv_content)
        write_csv(df_write)

        size_total = sum([sum((vv[v].stat().st_size for kk, vv in sample_info.items())) for v in shuffled_names])
        cnt_per_file = calc_count_per_file(sample_cnt=sample_cnt, size_total=size_total, file_cnt_max=tfrecord_num_max, size_per_file=tfrecord_size)

        # Write tfrecord
        def example_generator():

            for cur_name in shuffled_names:
                image = sample_info["image"][cur_name]
                label = sample_info["label"][cur_name]
                # dc.logger.info(image.resolve(), label.resolve())
                feature = {"path": _bytes_feature(str(image.resolve()).encode(encoding="utf-8")),
                           "image": _bytes_feature(image.read_bytes()),
                           "label": _bytes_feature(label.read_bytes())}
                if "mask" in sample_info:
                    mask = sample_info["mask"][cur_name]
                    feature["mask"] = _bytes_feature(mask.read_bytes())
                    # dc.logger.info(mask.resolve())
                yield tf.train.Example(features=tf.train.Features(feature=feature))

        write_tf_record(tf_record_dir=tfrecord_loc, cnt_per_file=cnt_per_file, example_generator=example_generator())

    elif format_type == FORMAT_SAME_DIR:
        assert task_type in [TASK_CLASSIFY, TASK_DETECT], "标住文件与图片在同一目录的数据集只能用于图像分类、目标检测任务。 当前任务类型：{}".format(task_type)
        assert "image_and_label" in data_detail
        content_path = Path(dataset_loc, data_detail["image_and_label"])
        assert content_path.exists() and content_path.is_dir()
        dc.logger.info("读取目录中文件信息： {}".format(content_path.resolve()))
        sample_info = dict()
        for p in content_path.iterdir():
            if p.is_file() and p.stat().st_size > 0 and (not p.name.startswith(".")):
                if p.stem not in sample_info:
                    sample_info[p.stem] = [0, 0, None, None]
                if p.suffix == ".xml":
                    if _get_label_from_xml(xml_file=p, label_path="./object/name") is None:
                        continue
                    sample_info[p.stem][1] += 1
                    sample_info[p.stem][3] = p
                else:
                    sample_info[p.stem][0] += 1
                    sample_info[p.stem][2] = p
            else:
                dc.logger.info("SKIP: {}".format(p.resolve()))
        dc.logger.info("筛选有效样本...")
        valid_samples = []
        for k, v in sample_info.items():
            if v[0] == v[1] == 1:
                valid_samples.append((k, v[2], v[3]))
        sample_cnt = len(valid_samples)
        dc.logger.info("有效样本数量：{}".format(len(valid_samples)))
        assert sample_cnt > 0

        random.shuffle(valid_samples)

        dc.logger.info("整理要输出的csv数据...")
        csv_content = {"path": [], "label": []}
        for name, img_path, label_path in valid_samples:
            csv_content["path"].append(str(img_path.resolve()))
            if task_type == TASK_CLASSIFY:
                csv_content["label"].append(_get_label_from_xml(label_path, "./object/name"))
            elif task_type in (TASK_DETECT, TASK_INSTANCE):
                csv_content["label"].append(";".join(parse_label(label_path)))
        df_write = pd.DataFrame(data=csv_content)
        write_csv(df_write)

        size_total = sum((v[1].stat().st_size + v[2].stat().st_size for v in valid_samples))
        cnt_per_file = calc_count_per_file(sample_cnt=sample_cnt, size_total=size_total, file_cnt_max=tfrecord_num_max, size_per_file=tfrecord_size)

        # Write tfrecord
        def example_generator():

            for _, image, label in valid_samples:
                # dc.logger.info(image.resolve(), label.resolve())
                feature = {"path": _bytes_feature(str(image.resolve()).encode(encoding="utf-8")),
                           "image": _bytes_feature(image.read_bytes()),
                           "label": _bytes_feature(label.read_bytes())}
                yield tf.train.Example(features=tf.train.Features(feature=feature))

        write_tf_record(tf_record_dir=tfrecord_loc, cnt_per_file=cnt_per_file, example_generator=example_generator())

    elif format_type == FORMAT_DIRS:
        assert task_type == TASK_CLASSIFY, "以文件夹作为类别的数据集只能用于图像分类任务。 当前任务类型：{}".format(task_type)
        assert "image_categories" in data_detail
        parent_dir = Path(dataset_loc, data_detail["image_categories"])
        assert parent_dir.exists() and parent_dir.is_dir()

        sample_info = dict()
        tfrecord_loc.mkdir(parents=True, exist_ok=True)
        for p in parent_dir.iterdir():
            if p.is_dir() and not p.samefile(tfrecord_loc):
                dc.logger.info("读取目录中文件信息： {}".format(p.resolve()))
                sample_info[p.name] = [pp for pp in p.iterdir() if pp.is_file() and pp.stat().st_size > 0 and (not pp.name.startswith("."))]

        dc.logger.info("筛选有效样本...")
        valid_samples = []
        for k, v in sample_info.items():
            if len(v) > 0:
                dc.logger.info("类别：{} 样本数：{}".format(k, len(v)))
                valid_samples.extend(((vv, k) for vv in v))
            else:
                dc.logger.info("类别：{} 无效".format(k))
        sample_cnt = len(valid_samples)
        dc.logger.info("总样本数：{}".format(sample_cnt))

        random.shuffle(valid_samples)

        dc.logger.info("整理要输出的csv数据...")
        csv_content = {"path": [], "label": []}
        for img_path, label in valid_samples:
            csv_content["path"].append(str(img_path.resolve()))
            csv_content["label"].append(label)
        df_write = pd.DataFrame(data=csv_content)
        write_csv(df_write)

        size_total = sum((v[0].stat().st_size for v in valid_samples))
        cnt_per_file = calc_count_per_file(sample_cnt=sample_cnt, size_total=size_total, file_cnt_max=tfrecord_num_max, size_per_file=tfrecord_size)

        # Write tfrecord
        def example_generator():
            xml_format = """<annotation>
        <filename>{}</filename>
        <object>
                <name>{}</name>
        </object>
</annotation>"""
            for image, label in valid_samples:
                # dc.logger.info(image.resolve(), label)
                feature = {"path": _bytes_feature(str(image.resolve()).encode(encoding="utf-8")),
                           "image": _bytes_feature(image.read_bytes()),
                           "label": _bytes_feature(xml_format.format(image.name, label).encode(encoding="utf-8"))}
                yield tf.train.Example(features=tf.train.Features(feature=feature))

        write_tf_record(tf_record_dir=tfrecord_loc, cnt_per_file=cnt_per_file, example_generator=example_generator())


    else:
        raise Exception("不支持当前的数据集格式：{}".format(format_type))
    end_time = datetime.now()
    dc.logger.info("CONVERT_TFRECORD 结束 ----- {} ----- 用时 {}s".format(end_time.isoformat(), (end_time - start_time).total_seconds()))


def create_tfrecord(dataset_path, tfrecord_output_path, csv_output_path, img_dir, label_xml_dir, mask_dir):
    dc.logger.info("START")

    # tfredorder转化需要的参数
    task_type = 'TASK_INSTANCE'
    format_type = 'FORMAT_VOC'
    tfrecord_num_max = 10
    tfrecord_size = 268435456
    data_detail = {"image":img_dir,"label":label_xml_dir,"mask":mask_dir}
    
    dataset_path = dataset_path
    tfrecord_output_path = tfrecord_output_path
    csv_output_path = csv_output_path

    print("dataset_path: ", dataset_path)
    print("tfrecord_output_path: ", tfrecord_output_path)
    print("csv_output_path: ", csv_output_path)
    print("img_dir: ", img_dir)
    print("label_xml_dir: ", label_xml_dir)
    print("mask_dir: ", mask_dir)
    
    
    # 3. 把数据集转换为tf-record文件
    convert_tfrecord(dataset_path=dataset_path,
                     task_type=task_type,
                     format_type=format_type,
                     data_detail=data_detail,
                     tfrecord_num_max=tfrecord_num_max,
                     tfrecord_size=tfrecord_size,
                     tfrecord_output_path=tfrecord_output_path,
                     csv_output_path=csv_output_path)


    dc.logger.info("generate tf-record success")
