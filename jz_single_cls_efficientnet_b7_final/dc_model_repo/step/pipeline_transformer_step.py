# -*- encoding: utf-8 -*-
from dc_model_repo.step.spark_step import SparkDCTransformer


class DataFrameMapper():

    def __init__(self, features=[]):
        self.features = features
        self._paramMap = {}


class PipelineTransformerStep(SparkDCTransformer):
    """控制Step的个数,解决Step目录太多的问题。
    """

    def __init__(self, operator, algorithm_name, extension=None, **kwargs):
        from pyspark.ml import Pipeline
        pipeline_model = Pipeline(stages=operator.features)

        self.operator = pipeline_model
        self.model_path = 'data/model'
        self._params = None
        self.spark_df_schema = None

        # 从model中提取 input_cols
        super(PipelineTransformerStep, self).__init__(operator=self.operator,
                                                      algorithm_name=algorithm_name, extension=extension, **kwargs)
        if algorithm_name is None:
            step = operator.features[len(operator.features) - 1]
            self.algorithm_name = step.__class__.__name__
        else:
            self.algorithm_name = algorithm_name

        # Fixed by zk. 2020.11.25. 现在DCStep在fit时候会校验X的列是否全部包含初始化时的input_cols，如果有些列X中没有会报错，
        # spark算子组成的pipeline中间生成的临时列如果加到input_cols中，会引发报错。
        # 由于spark串联之后基本都是第一个算子的列为原始数据的输入列，后续算子使用的列为前面算子生成的临时列
        # 所以这里改为只把spark pipeline中的第一个算子的输入列当作input_cols
        first_step = operator.features[0]
        self.input_cols = super(PipelineTransformerStep, self).get_input_cols(first_step)

    def persist_model(self, fs, destination):

        # 将Spark的Transformer写到hdfs上的data/model目录
        p_model = self.serialize_data_path(destination) + "/model"
        self.model.save(p_model)

    def prepare(self, path, **kwargs):
        """
        从HDFS上加载Step。
        :param path:
        :param kwargs:
        :return:
        """
        from dc_model_repo.base.mr_log import logger
        # 1. 使用Pipeline加载模型
        p_model = path + "/data/model"
        from pyspark.ml import PipelineModel
        self.model = PipelineModel.load(p_model)
        logger.info("成功加载Spark模型: %s" % str(self.model))

    def predict(self, x):
        return self.transform(x)

    def transform_data(self, X, **kwargs):
        return self.model.transform(X)
