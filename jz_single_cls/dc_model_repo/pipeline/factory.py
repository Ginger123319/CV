# -*- encoding: utf-8 -*-


def collect_pipeline(**kwargs):
    """在工作流环境搜集 step 信息，组装 :class:`dc_model_repo.pipeline.pipeline.DCPipeline`

    1. 调试模式不生成模型
    2. 从Pipes接口中获取 DAG
    3. 解析模块信息
    4. 根据Step的地址反序列化得到Steps
    5. 剔除最后一个生成Pipeline之前的 Step
    6. 如果最后一个estimator相邻有连续的estimator，只保留最后一个
    7. Pipeline的名称（Pipes 需要提前知道模型的名称，所以约定为pipeline_<block_id>）
    8. 构建Pipeline:

        1. 解析出estimatorBlockId(不能在Pipeline init方法中处理。)
        2. 查找 PipelineInitDCStep
        3. 移除 PipelineInitDCStep
        4. 从 PipelineInitDCStep 解析出Pipeline的 输入特征、input_type、样本数据、label_mapping。
        5. 设置各个Step的路径

    Args:
        **kwargs: 传给实际的step collector。
          目前可用参数：

            - name 模型名称，可选，默认为 `pipeline_{pipeline.id}`。
            - learning_type 模型类型，见 :class:`dc_model_repo.base.LearningType`。
            - performance 模型性能，为 :class:`dc_model_repo.base.ChartData` 类型数组。或None。
            - performance_file 模型评估附件地址，字符串类型文件路径。
              路径可以是名为performance.json的文件或者包含附件的文件夹路径。
              当performance不为None时，不再对performance.json文件进行解析。

    Returns:
        :class:`dc_model_repo.pipeline.pipeline.DCPipeline`

    Raises:
        Exception: 无法找到step collector时报错

    """
    from dc_model_repo.base import collector_manager
    step_collector = collector_manager.get_step_collector()
    if step_collector:
        return step_collector.collect_pipeline(**kwargs)
    else:
        raise Exception("NullPointException")
