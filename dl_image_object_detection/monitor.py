import json
import pandas as pd
from pathlib import Path
from datacanvas.aps import dc
from utils import get_pipeline, get_df, evaluate
from workflow_collector.workflow_collector import RuntimeProxy

work_dir = Path(dc.conf.global_params.work_dir, dc.conf.global_params.block_id)
Path(work_dir).mkdir(parents=True, exist_ok=True)
work_dir = str(work_dir)

dc.logger.info("dc is ready!")

is_debug_model = RuntimeProxy().is_debug_model()
dc.logger.info("is_debug_model: {}".format(repr(is_debug_model)))

# # 读入参数
# model_uris = ['model://aps_published-00000000-aaaa-0000-000a-000000000001-31c5016c-e121-4e67-bb99-73104a4d54ef/cb12c674-0de7-468d-bfa6-cb0d89bd363f']
# test_data = str(dc.conf.inputs.test_data)
# output = work_dir

# 解析参数
model_uris = dc.conf.input.model.ids  # list[]类型，获取模型的uri，跑批及评估时只有一个，监控时为1个或多个
model_metric = dc.conf.input.model.metric  # list[]类型，跑批是为None，评估时为全量，监控时为选择的评估项
output = dc.conf.output  # str 获取要输出到的目录

# 读入数据
# ds = dc.dataset(test_data).read()
ds_dir = dc.datasource(dc.conf.input.data.source)
ds = ds_dir.read_dir(**dc.conf.input.data.schema)

test_data = get_df(ds)
test_data = test_data.reset_index(drop=True)
x_test = test_data[['path']]

for model_uri in model_uris:
    pipeline = get_pipeline(work_dir, model_uri, is_debug_model)
    # 模型预测
    predictions = pipeline.predict(x_test)
    predictions = pd.concat([test_data, predictions], axis=1)

    # 评估
    performance = evaluate(predictions, work_dir)

    # 输出结果
    with open(output + '/performance_{}.json'.format(pipeline.id), 'w') as f:
        json.dump(performance, f)

dc.logger.info("Done!")
