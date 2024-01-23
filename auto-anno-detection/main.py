import pandas as pd
import utils
# 参数
# input_label_path
#     已标注数据文件路径(原始不包括难例筛选后确认数据)
#     /{workdir}/smartLabelJob/{smartLabelJobId}/input/input_label.csv
# input_added_label_path
#     所有之前难例筛选后确认数据（累计）
#     /{workdir}/smartLabelJob/{smartLabelJobId}/input/input_added_label.csv
# input_unlabel_path
#     未标注数据文件
#     /{workdir}/smartLabelJob/{smartLabelJobId}/input/input_unlabel.csv
# model_url
#     预训练模型地址
#     当 is_first_train 为ture是 model://reponame/modelId ;为flase 时 /{workdir}/smartLabelJob/{lastSmartLabelJobId}/model/{uuid}
# result_path
#     预测文件路径
#     /{workdirmodel}/smartLabelJob/{smartLabelJobId}/result/result.csv
# target_model_Id
#     生成的模型id
#     {smartLabelJobId}
# target_model_path
#     生成的模型保存路径
#     /{workdir}/smartLabelJob/{smartLabelJobId}/model/{smartLabelJobId
# labe_type
#     智能标注类型
#     图片单分类101,图片多分类102,图片目标检测103 ，201 文本单分类，202 文本多分类，203 文件实体抽取，205文情感倾向分析
# is_first_train
#     是否首次训练
#     true/flase

input_label_path = "workdir/smartLabelJob/smartLabelJobId/input/input_label.csv"
input_added_label_path = "workdir/smartLabelJob/smartLabelJobId/input/input_added_label.csv"
input_unlabel_path = "workdir/smartLabelJob/smartLabelJobId/input/input_unlabel.csv"
model_url = "workdir/smartLabelJob/smartLabelJobId/model/prev_model_id"
result_path = "workdir/smartLabelJob/smartLabelJobId/result/result.csv"
target_model_Id = "model_id"
target_model_path = "workdir/smartLabelJob/smartLabelJobId/model/model_id"
is_first_train = True

hyp = {}
opt = {}

work_dir = "workdir/smartLabelJob/smartLabelJobId/"

df_init = pd.read_csv(input_label_path)
df_anno = pd.read_csv(input_added_label_path) if is_first_train else None
df_img = pd.read_csv(input_unlabel_path)

# download pipeline
# load and prepare pipeline
load_pipeline = True
if load_pipeline:
    from utils import load_pipeline_model

    model = load_pipeline_model(model_url=model_url, work_dir=work_dir)
    from object_detect import Model

    assert isinstance(model, Model)
    model.adjust_model()
else:
    from object_detect.my_model import MyModel

    model = MyModel()

# split train val
df_train, df_val = utils.split_train_val(df_init, df_anno, is_first_train, val_size=0.2)

# train
model.train_model(df_train=df_train, df_val=df_val)

# predict
df_pred = model.predict(df_img)

# select
df_result = utils.select_hard_example(df_pred)

# save result csv
df_result.to_csv(result_path)

# persist pipeline
utils.save_pipeline_model(model=model, model_id=target_model_Id, save_dir=target_model_path)
