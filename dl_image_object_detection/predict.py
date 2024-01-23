import zipfile
import pandas as pd
from pathlib import Path
from datacanvas.aps import dc
from dc_model_repo import model_repo_client
from dc_model_repo.pipeline.pipeline import DCPipeline
from workflow_collector.workflow_collector import RuntimeProxy


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


def to_df_voc(cur_ds):
    content = {"path": []}
    for img in cur_ds.data:
        if check_img(img.data):
            content["path"].append(img.data)
    df = pd.DataFrame(content)
    return df


def to_df_coco(cur_coco):
    content = {"path": []}
    for img_id in cur_coco.getImgIds():
        img = cur_coco.imgs[img_id]
        img_path = img["file_full_path"]
        if check_img(img_path):
            content["path"].append(img_path)
    df = pd.DataFrame(content)
    return df


def to_df_unlabeled(cur_ds):
    content = {"path": []}
    for img in cur_ds.data:
        if check_img(img.data):
            content["path"].append(img.data)
    df = pd.DataFrame(content)
    return df


def get_df(cur_ds):
    valid_ds(cur_ds)

    if cur_ds.label_format == "VOC":
        df_all = to_df_voc(cur_ds)
    elif cur_ds.label_format == "COCO":
        df_all = to_df_coco(cur_ds.data)
    else:
        df_all = to_df_unlabeled(cur_ds)

    print("样本条数：", len(df_all["path"]))
    return df_all


# 解压模型文件
def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


work_dir = Path(dc.conf.global_params.work_dir, dc.conf.global_params.block_id)
Path(work_dir).mkdir(parents=True, exist_ok=True)
work_dir = str(work_dir)

dc.logger.info("dc is ready!")
is_debug_model = RuntimeProxy().is_debug_model()
dc.logger.info("is_debug_model: {}".format(repr(is_debug_model)))

# # 读入参数
# model_uri = 'model://aps_published-00000000-aaaa-0000-000a-000000000001-31c5016c-e121-4e67-bb99-73104a4d54ef/6b8055f8-04c9-4040-b54e-c448c8e8335f'
# test_data = str(dc.conf.inputs.test_data)
# output = work_dir

# 解析参数
model_uris = dc.conf.input.model.ids  # list[]类型，获取模型的uri，跑批及评估时只有一个，监控时为1个或多个
model_metric = dc.conf.input.model.metric  # list[]类型，跑批是为None，评估时为全量，监控时为选择的评估项
output = dc.conf.output  # str 获取要输出到的目录
model_uri = model_uris[0]

# 读入数据
ds_dir = dc.datasource(dc.conf.input.data.source)
ds = ds_dir.read_dir(**dc.conf.input.data.schema)
# ds = dc.dataset(test_data).read()
x_test = get_df(ds)

# 读入模型
model_tmp_path = work_dir + "/model.zip"
model_path = work_dir + "/model"
model_repo_client.get(model_uri, model_tmp_path, timeout=(2, 60))

unzip_file(model_tmp_path, model_path)

# 加载模型
dc.logger.info("model_path:", model_path)
pipeline = DCPipeline.load(model_path, 'local', debug_log=is_debug_model)
pipeline.prepare()

# 模型预测
predictions = pipeline.predict(x_test)
predictions = pd.concat([x_test, predictions], axis=1)

# prediction
predictions.to_csv(output + '/prediction_{}.csv'.format(pipeline.id), index=False)

dc.logger.info("Done!")
