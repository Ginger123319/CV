# -*- encoding: utf-8 -*-

from dc_model_repo.base import FrameworkType, StepType, Param, Field, Output, ModelFileFormatType, LearningType
from dc_model_repo.step.base import ModelWrapperDCStep, BaseEstimator, BaseTransformer
from dc_model_repo.util import str_util, cls_util, validate_util, operator_output_util
import abc
import six
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import PipelineModel
# from pyspark.ml.base import HasLabelCol, HasPredictionCol, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, Params
from pyspark.sql.functions import udf

from pyspark.sql.types import StringType, DoubleType, ArrayType


class SparkDCTransformer(BaseTransformer):

    def __init__(self, operator, algorithm_name, extension=None, **kwargs):
        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)

        # 从model中提取 input_cols
        super(SparkDCTransformer, self).__init__(operator=operator,
                                                 framework=FrameworkType.Spark,
                                                 model_format=ModelFileFormatType.DIR,
                                                 input_cols=SparkDCTransformer.get_input_cols(operator),
                                                 algorithm_name=algorithm_name,
                                                 extension=extension,
                                                 **kwargs)

        self.model_path = 'data/model'
        self.spark_df_schema = None

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator"]

    @staticmethod
    def get_input_cols(model):
        col = SparkDCTransformer.__get_value_in_model__(["inputCol", "inputCols", "featuresCol"], model)
        if col is None:
            return ['features', 'label']
        else:
            return col

    @staticmethod
    def __get_value_in_model__(keys, obj):
        if obj is None:
            return None
        _items_param_map = obj._paramMap.items()
        for k in keys:
            for item in _items_param_map:
                if str_util.to_str(item[0].name) == k:
                    v = item[1]
                    if isinstance(v, list):
                        return v
                    else:
                        return [v]
        return None

    def transform_data(self, X, **kwargs):
        return self.model.transform(X)

    def persist_model(self, fs, destination):
        # 将Spark的Transformer写到hdfs上的data/model目录
        p_model = self.serialize_data_path(destination) + "/model"
        pipeline_model = PipelineModel(stages=[self.model])
        pipeline_model.save(p_model)

    def prepare(self, path, **kwargs):
        """
        从HDFS上加载Step。
        :param path:
        :param kwargs:
        :return:
        """
        from dc_model_repo.base.mr_log import logger
        # 1. 使用Pipeline加载模型
        p_model = path + "/" + self.model_path
        self.model = PipelineModel.load(p_model).stages[0]
        logger.info("成功加载Spark模型: %s" % str(self.model))

    def get_params(self):
        _items_param_map = self.operator._paramMap.items()
        params = []
        for item in _items_param_map:
            p_name = str_util.to_str(item[0].name)
            p_value = str_util.to_str(item[1])
            params.append(Param(p_name, None, p_value))
        return params

    def fit_post(self, X, y=None, options=None, **kwargs):
        self.spark_df_schema = X.schema


class SparkDCEstimator(BaseEstimator):

    def __init__(self, operator, algorithm_name=None, extension=None, **kwargs):
        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)

        output_cols = kwargs.get("output_cols", None)
        if output_cols is None and isinstance(operator, HasPredictionCol):
            output_cols = [operator.getPredictionCol()]
        target_cols = kwargs.get("target_cols", None)
        if target_cols is None and isinstance(operator, HasLabelCol):
            target_cols = [operator.getLabelCol()]

        learning_type = kwargs.get("learning_type", None)
        if learning_type is None and type(operator).__module__ == "pyspark.ml.clustering":
            learning_type = LearningType.Clustering

        # 从model中提取 input_cols
        super(SparkDCEstimator, self).__init__(operator=operator,
                                               framework=FrameworkType.Spark,
                                               model_format=ModelFileFormatType.DIR,
                                               input_cols=SparkDCEstimator._get_input_cols(operator),
                                               algorithm_name=algorithm_name,
                                               extension=extension,
                                               output_cols=output_cols,
                                               target_cols=target_cols,
                                               learning_type=learning_type,
                                               **kwargs)
        self.model_path = 'data/model'
        self.spark_df_schema = None
        self.label_indexer_model = None
        self.labels = None  # 必须是list类型，不能是 ndarray

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator", "label_indexer_model"]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        """获取样本数据。
        Returns:
        """
        # 生成Target样本数据, 注意： 必须在LabelEncoder之前。
        label_col = self.get_label_col()
        return None if label_col is None else self.get_data_sampler(x.select([label_col]))

    @staticmethod
    def _get_input_cols(model):
        if model is None:
            return None
        assert isinstance(model, Params)
        _items_param_map = model._defaultParamMap.copy()
        _items_param_map.update(model._paramMap)
        for item in _items_param_map.items():
            if str_util.to_str(item[0].name) in ["inputCol", "inputCols", "featuresCol"]:
                v = item[1]
                if isinstance(v, list):
                    return v
                else:
                    return [v]
        return ['features', 'label']

    def persist_model(self, fs, destination):
        # 将Spark的Transformer写到hdfs上的data/model目录
        p_model = self.serialize_data_path(destination) + "/model"
        pipeline_model = PipelineModel(stages=[self.model])
        pipeline_model.save(p_model)

    def prepare(self, path, **kwargs):
        """
        从HDFS上加载Step。
        :param path:
        :param kwargs:
        :return:
        """
        from dc_model_repo.base.mr_log import logger
        # 1. 使用Pipeline加载模型
        p_model = path + "/" + self.model_path
        self.model = PipelineModel.load(p_model).stages[0]
        logger.info("成功加载Spark模型: %s" % str(self.model))

    def persist_explanation(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        explanation = []

        # 1. 从训练后的模型解析特征重要性
        if hasattr(self.model, 'featureImportances'):

            # 1.1. 读取数据列和特征重要性
            feature_importances = self.model.featureImportances.toArray()

            # 1.2. 读取训练列
            # todo 传递过来数量正确的train_columns
            # if self.train_columns is None:
            origin_cols = ["x%s" % i for i in range(len(feature_importances))]
            # else:
            #    origin_cols = self.train_columns

            # 1.3. 校验列是否正确
            if len(origin_cols) == len(feature_importances):
                cd = self.get_feature_importances(origin_cols, feature_importances, 'string')
                explanation.append(cd)
            else:
                raise Exception("用户设置的'origin_cols'和模型中的列数不同, 设置的列的个数:%s,模型中列的个数: %d " % (len(origin_cols), len(feature_importances)))

            # 1.4. 写入数据
            self.persist_explanation_object(fs, destination, explanation)
        else:
            logger.info("当前模型中没有特征重要性的数据。")

    def predict(self, X, **kwargs):
        return self.transform_data(X, **kwargs)

    def transform_data(self, X, **kwargs):
        """转换数据。

        Args:
            X:
            labels: 不为空时，将预测结果反转成labels中的数据，概率列的列名也将使用labels的数据。
            **kwargs:

        Returns:

        """
        from dc_model_repo.base.mr_log import logger
        # 1. 预测
        result_df = self.model.transform(X)
        output = self.outputs[0]
        output_col = output.name

        # 2. 当 labels 不为空才对结果进行反转
        if hasattr(self, 'labels') and validate_util.is_non_empty_list(self.labels):  # 兼容3.2 版本没有labels属性
            logger.info("参数 labels 不为空，将对预测结果列进行反转成字符串。")
            # 2.1. 反转生成为 prediction_reversed 列
            reverse_output_col_name = output_col + "_reversed"
            index2str = IndexToString(inputCol=output_col, outputCol=reverse_output_col_name, labels=self.labels)
            result_df = index2str.transform(result_df)

            # 2.2. prediction列重命名为raw_prediction，再把prediction_reversed命名为prediction。
            result_df = result_df.withColumnRenamed(output_col, "raw_" + output_col) \
                .withColumnRenamed(reverse_output_col_name, output_col)

        # 3. 原始概率列转换数组(如果有概率`model.transform`就会生成概率结果)
        probability_array_col_name = "probability_array"  #
        if output.raw_prob_field is not None:
            # 3.2. 创建转换概率成数组的udf函数
            def to_array(col):
                return udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))(col)

            raw_prob_field_name = output.raw_prob_field.name
            logger.info("概率结果转换成数组, 概率列名为：%s" % raw_prob_field_name)
            result_df = result_df.withColumn(probability_array_col_name, to_array(raw_prob_field_name))

        # 4. 生成最大概率列
        if output.max_prob_field is not None and output.raw_prob_field is not None:
            # 4.1. 创建求最大概率列函数
            def calc_max_func(v):
                return max(v)

            calc_max_udf = udf(calc_max_func, DoubleType())

            # 4.2. 计算数据
            result_df = result_df.withColumn(output.max_prob_field.name, calc_max_udf(probability_array_col_name))

        # 5. 生成标签概率列
        label_prob_fields = output.label_prob_fields
        if validate_util.is_non_empty_list(label_prob_fields):
            # 5.1. 拆分概率udf
            # 注意： 列名中不能有"-"， 需要用`` 引用列名
            # 概率取值的问题： 如果分类是10,20,30 会有31个标签，numClass是31，标签概率列也会生成31个。
            prob_select_expr = [str('%s[%s] as `%s`' % (probability_array_col_name, i, f.name)) for i, f in enumerate(label_prob_fields)]
            logger.info("拆分label概率表达式：%s" % ",".join(prob_select_expr))
            result_df = result_df.selectExpr("*", *prob_select_expr)

        return result_df

    def get_targets(self, X, y=None, options=None, **kwargs):
        """获取训练目标列的信息, 仅estimator时候需要重写。
        Returns:
            list(dc_model_repo.base.Feature)
        """
        label_col = self.get_label_col()
        if label_col is None:
            # 如果为聚类，可以没有label
            return None

        label_data_type = self.get_data_type(X, label_col)

        return [Field(label_col, label_data_type)]

    def get_label_col(self):
        estimator = self.get_estimator()
        label_col = str_util.to_str(estimator.getLabelCol()) if isinstance(estimator, HasLabelCol) else None
        return label_col

    def get_prediction_col(self):
        estimator = self.get_estimator()
        predict_col = str_util.to_str(estimator.getPredictionCol()) if isinstance(estimator, HasPredictionCol) else None
        return predict_col

    def get_data_type(self, df, col):
        return df.select([col]).dtypes[0][1]

    def get_outputs(self, X, y=None, options=None, **kwargs):
        """解析模型的输出信息列信息。

        输出结果：prediction
        原始概率: probability
        标签概率: probability_proba_{label}

        Returns:
        """
        from dc_model_repo.base.mr_log import logger

        # 1. 解析概率
        max_prob_field = None
        raw_prob_field = None
        label_prob_fields = None

        if hasattr(self.get_estimator(), 'getProbabilityCol'):
            probability_col = self.get_estimator().getProbabilityCol()
            # 分类模型会有概率，有概率就会有标签的概率
            if hasattr(self.model, 'numClasses') and self.model.numClasses > 1:

                # 1.1. 生成原始概率列
                raw_prob_field = Field(probability_col, 'double')

                # 1.2. 生成最大概率列
                max_prob_field = operator_output_util.get_max_prob_field(probability_col, 'double')

                # 1.3. 更新全概率列shape
                num_classes = self.model.numClasses
                raw_prob_field.shape = (1, num_classes)

                # 1.4. 设置标签的概率(只有在分类并且有概率时候才有)
                if validate_util.is_non_empty_list(self.labels):
                    if len(self.labels) == num_classes:
                        # 1.4.1. 如果是string类型的label，用原来的字符串名称作为概率列的后缀名称
                        label_prob_fields = operator_output_util.get_label_prob_fields(probability_col, self.labels, 'double')
                    else:
                        # 1.4.2. 概率列信息重置为空
                        logger.warning("模型训练时StringIndexer中labels的长度=%d，而模型中class的数量=%d，不一致，不能生成概率列。")  # 可能为程序训练的bug或者用户设置错误，发生时容错
                        max_prob_field = None
                        raw_prob_field = None
                        label_prob_fields = None
                else:
                    # 1.4.3. 如果是用double类型的label训练的，用数字后缀作为概率列的名称
                    label_prob_fields = operator_output_util.get_label_prob_fields(probability_col, range(num_classes), 'double')

        # 2. 输出列名称
        output_name = self.get_estimator().getPredictionCol()

        # 3. 获取Label列的类型
        if self.get_label_col() is None:
            output_data_type = "unknown"
        else:
            output_data_type = self.get_data_type(X, self.get_label_col())

        output = Output(name=output_name,
                        type=output_data_type,
                        shape=None,
                        max_prob_field=max_prob_field,
                        raw_prob_field=raw_prob_field,
                        label_prob_fields=label_prob_fields)

        return [output]

    def get_estimator(self):
        return self.operator

    def fit_model(self, X, y=None, options=None, **kwargs):
        X = self.do_string_indexer_on_label(X)
        return super(SparkDCEstimator, self).fit_model(X, y, options, **kwargs)

    def do_string_indexer_on_label(self, X):
        """对String类型的label列进行编码。
        Args:
            X:

        Returns:

        """
        label_col = self.get_label_col()
        if label_col is None:
            return X

        label_data_type = self.get_data_type(X, label_col)

        from dc_model_repo.base.mr_log import logger

        # 1. 当标签列为字符串类型时候，做StringIndexer
        if label_data_type == 'string':
            logger.info("当前为分布式模型，并且label列为string类型，对该列进行编码。")
            self.label_indexer_model = StringIndexer(inputCol=label_col, outputCol=self.get_indexed_label_col()).fit(X)
            # 1.1. 记录下labels
            self.labels = [str_util.to_str(la) for la in self.label_indexer_model.labels]

            # 1.2. 新生成编码后的列{label}_indexed，并删除{label}列
            X = self.label_indexer_model.transform(X).drop(label_col)

            # 1.3. {label}_indexed列重命名为 {label}列
            X = X.withColumnRenamed(self.get_indexed_label_col(), label_col)
        # fit 之后需要把string类型进行还原
        return X

    def get_indexed_label_col(self):
        # 约定编码后的label列，该列最终会被重命名。
        return self.get_label_col() + "_indexed"

    def get_params(self):
        """解析Spark模型的参数， 使用训练前的原始模型。
        Returns:
        """
        return self.get_params_from_dict_items(self.operator.__dict__, self.input_cols)


class SparkDCTuningEstimator(BaseEstimator):

    def __init__(self, operator, algorithm_name=None, extension=None, **kwargs):

        # 0. 校验参数
        from pyspark.ml.tuning import CrossValidator, TrainValidationSplit
        assert isinstance(operator, (CrossValidator, TrainValidationSplit))
        estimator = operator.getEstimator()

        # 1. 设置调优对象
        self.tuning_estimator = operator

        # 2. 覆盖父类生成的算法名称
        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(self.get_estimator())

        # 3. 获取输出列和label列名称
        output_cols = kwargs.get("output_cols", None)
        if output_cols is None and isinstance(estimator, HasPredictionCol):
            output_cols = [estimator.getPredictionCol()]
        target_cols = kwargs.get("target_cols", None)
        if target_cols is None and isinstance(estimator, HasLabelCol):
            target_cols = [estimator.getLabelCol()]

        learning_type = kwargs.get("learning_type", None)
        if learning_type is None and type(estimator).__module__ == "pyspark.ml.clustering":
            learning_type = LearningType.Clustering

        # 3. 调用父类构造方法
        # 从model中提取 input_cols
        input_cols = SparkDCTuningEstimator._get_input_cols(operator.getOrDefault(operator.estimator))
        super(SparkDCTuningEstimator, self).__init__(operator=operator,
                                                     framework=FrameworkType.Spark,
                                                     model_format=ModelFileFormatType.DIR,
                                                     input_cols=input_cols,
                                                     algorithm_name=algorithm_name,
                                                     extension=extension,
                                                     output_cols=output_cols,
                                                     target_cols=target_cols,
                                                     learning_type=learning_type,
                                                     **kwargs)
        self.model_path = 'data/model'
        self.spark_df_schema = None
        self.label_indexer_model = None
        self.labels = None  # 必须是list类型，不能是 ndarray

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator", "label_indexer_model"]


    @staticmethod
    def _get_input_cols(model):
        if model is None:
            return None
        assert isinstance(model, Params)
        _items_param_map = model._defaultParamMap.copy()
        _items_param_map.update(model._paramMap)
        for item in _items_param_map.items():
            if str_util.to_str(item[0].name) in ["inputCol", "inputCols", "featuresCol"]:
                v = item[1]
                if isinstance(v, list):
                    return v
                else:
                    return [v]
        return ['features', 'label']

    def persist_model(self, fs, destination):
        # 将Spark的Transformer写到hdfs上的data/model目录
        p_model = self.serialize_data_path(destination) + "/model"
        pipeline_model = PipelineModel(stages=[self.model])
        pipeline_model.save(p_model)

    def prepare(self, path, **kwargs):
        """
        从HDFS上加载Step。
        :param path:
        :param kwargs:
        :return:
        """
        from dc_model_repo.base.mr_log import logger
        # 1. 使用Pipeline加载模型
        p_model = path + "/" + self.model_path
        self.model = PipelineModel.load(p_model).stages[0]
        logger.info("成功加载Spark模型: %s" % str(self.model))

    def persist_explanation(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        explanation = []

        # 1. 从训练后的模型解析特征重要性
        if hasattr(self.model, 'featureImportances'):

            # 1.1. 读取数据列和特征重要性
            feature_importances = self.model.featureImportances.toArray()

            # 1.2. 读取训练列
            # todo 传递过来数量正确的train_columns
            # if self.train_columns is None:
            origin_cols = ["x%s" % i for i in range(len(feature_importances))]
            # else:
            #    origin_cols = self.train_columns

            # 1.3. 校验列是否正确
            if len(origin_cols) == len(feature_importances):
                cd = self.get_feature_importances(origin_cols, feature_importances, 'string')
                explanation.append(cd)
            else:
                raise Exception("用户设置的'origin_cols'和模型中的列数不同, 设置的列的个数:%s,模型中列的个数: %d " % (len(origin_cols), len(feature_importances)))

            # 1.4. 写入数据
            self.persist_explanation_object(fs, destination, explanation)
        else:
            logger.info("当前模型中没有特征重要性的数据。")

    def predict(self, X, **kwargs):
        return self.transform_data(X, **kwargs)

    def transform_data(self, X, **kwargs):
        """转换数据。

        Args:
            X:
            labels: 不为空时，将预测结果反转成labels中的数据，概率列的列名也将使用labels的数据。
            **kwargs:

        Returns:

        """
        from dc_model_repo.base.mr_log import logger
        # 1. 预测
        result_df = self.model.transform(X)
        output = self.outputs[0]
        output_col = output.name

        # 2. 当 labels 不为空才对结果进行反转
        if hasattr(self, 'labels') and validate_util.is_non_empty_list(self.labels):  # 兼容3.2 版本没有labels属性
            logger.info("参数 labels 不为空，将对预测结果列进行反转成字符串。")
            # 2.1. 反转生成为 prediction_reversed 列
            reverse_output_col_name = output_col + "_reversed"
            index2str = IndexToString(inputCol=output_col, outputCol=reverse_output_col_name, labels=self.labels)
            result_df = index2str.transform(result_df)

            # 2.2. prediction列重命名为raw_prediction，再把prediction_reversed命名为prediction。
            result_df = result_df.withColumnRenamed(output_col, "raw_" + output_col) \
                .withColumnRenamed(reverse_output_col_name, output_col)

        # 3. 原始概率列转换数组(如果有概率`model.transform`就会生成概率结果)
        probability_array_col_name = "probability_array"  #
        if output.raw_prob_field is not None:
            # 3.2. 创建转换概率成数组的udf函数
            def to_array(col):
                return udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))(col)

            raw_prob_field_name = output.raw_prob_field.name
            logger.info("概率结果转换成数组, 概率列名为：%s" % raw_prob_field_name)
            result_df = result_df.withColumn(probability_array_col_name, to_array(raw_prob_field_name))

        # 4. 生成最大概率列
        if output.max_prob_field is not None and output.raw_prob_field is not None:
            # 4.1. 创建求最大概率列函数
            def calc_max_func(v):
                return max(v)

            calc_max_udf = udf(calc_max_func, DoubleType())

            # 4.2. 计算数据
            result_df = result_df.withColumn(output.max_prob_field.name, calc_max_udf(probability_array_col_name))

        # 5. 生成标签概率列
        label_prob_fields = output.label_prob_fields
        if validate_util.is_non_empty_list(label_prob_fields):
            # 5.1. 拆分概率udf
            # 注意： 列名中不能有"-"， 需要用`` 引用列名
            # 概率取值的问题： 如果分类是10,20,30 会有31个标签，numClass是31，标签概率列也会生成31个。
            prob_select_expr = [str('%s[%s] as `%s`' % (probability_array_col_name, i, f.name)) for i, f in enumerate(label_prob_fields)]
            logger.info("拆分label概率表达式：%s" % ",".join(prob_select_expr))
            result_df = result_df.selectExpr("*", *prob_select_expr)

        return result_df

    def get_targets(self, X, y=None, options=None, **kwargs):
        """获取训练目标列的信息, 仅estimator时候需要重写。
        Returns:
            list(dc_model_repo.base.Feature)
        """
        label_col = self.get_label_col()
        if label_col is None:
            # 如果为聚类，可以没有label
            return None

        label_data_type = self.get_data_type(X, label_col)

        return [Field(label_col, label_data_type)]

    def get_label_col(self):
        estimator = self.get_estimator()
        label_col = str_util.to_str(estimator.getLabelCol()) if isinstance(estimator, HasLabelCol) else None
        return label_col

    def get_prediction_col(self):
        estimator = self.get_estimator()
        predict_col = str_util.to_str(estimator.getPredictionCol()) if isinstance(estimator, HasPredictionCol) else None
        return predict_col

    def get_data_type(self, df, col):
        return df.select([col]).dtypes[0][1]

    def get_outputs(self, X, y=None, options=None, **kwargs):
        """解析模型的输出信息列信息。

        输出结果：prediction
        原始概率: probability
        标签概率: probability_proba_{label}

        Returns:
        """
        from dc_model_repo.base.mr_log import logger

        # 1. 解析概率
        max_prob_field = None
        raw_prob_field = None
        label_prob_fields = None

        if hasattr(self.get_estimator(), 'getProbabilityCol'):
            probability_col = self.get_estimator().getProbabilityCol()
            # 分类模型会有概率，有概率就会有标签的概率
            if hasattr(self.model, 'numClasses') and self.model.numClasses > 1:

                # 1.1. 生成原始概率列
                raw_prob_field = Field(probability_col, 'double')

                # 1.2. 生成最大概率列
                max_prob_field = operator_output_util.get_max_prob_field(probability_col, 'double')

                # 1.3. 更新全概率列shape
                num_classes = self.model.numClasses
                raw_prob_field.shape = (1, num_classes)

                # 1.4. 设置标签的概率(只有在分类并且有概率时候才有)
                if validate_util.is_non_empty_list(self.labels):
                    if len(self.labels) == num_classes:
                        # 1.4.1. 如果是string类型的label，用原来的字符串名称作为概率列的后缀名称
                        label_prob_fields = operator_output_util.get_label_prob_fields(probability_col, self.labels, 'double')
                    else:
                        # 1.4.2. 概率列信息重置为空
                        logger.warning("模型训练时StringIndexer中labels的长度=%d，而模型中class的数量=%d，不一致，不能生成概率列。")  # 可能为程序训练的bug或者用户设置错误，发生时容错
                        max_prob_field = None
                        raw_prob_field = None
                        label_prob_fields = None
                else:
                    # 1.4.3. 如果是用double类型的label训练的，用数字后缀作为概率列的名称
                    label_prob_fields = operator_output_util.get_label_prob_fields(probability_col, range(num_classes), 'double')

        # 2. 输出列名称
        output_name = self.get_estimator().getPredictionCol()

        # 3. 获取Label列的类型
        if self.get_label_col() is None:
            output_data_type = "unknown"
        else:
            output_data_type = self.get_data_type(X, self.get_label_col())

        output = Output(name=output_name,
                        type=output_data_type,
                        shape=None,
                        max_prob_field=max_prob_field,
                        raw_prob_field=raw_prob_field,
                        label_prob_fields=label_prob_fields)

        return [output]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        """获取样本数据。
        Returns:
        """
        # 生成Target样本数据, 注意： 必须在LabelEncoder之前。
        label_col = self.get_label_col()
        return None if label_col is None else self.get_data_sampler(x.select([label_col]))

    def fit_post(self, X, y=None, options=None, **kwargs):
        """分布式Estimator训练。
        Args:
            options:
            X:
            y:
            **kwargs: 可选train_cols, 用于生成特征重要性。
        Returns:
        """
        # 1. 训练列名
        self.train_columns = kwargs.get('train_cols')

        return self

    def do_string_indexer_on_label(self, X):
        """对String类型的label列进行编码。
        Args:
            X:

        Returns:

        """
        label_col = self.get_label_col()
        if label_col is None:
            return X

        label_data_type = self.get_data_type(X, label_col)

        from dc_model_repo.base.mr_log import logger

        # 1. 当标签列为字符串类型时候，做StringIndexer
        if label_data_type == 'string':
            logger.info("当前为分布式模型，并且label列为string类型，对该列进行编码。")
            self.label_indexer_model = StringIndexer(inputCol=label_col, outputCol=self.get_indexed_label_col()).fit(X)
            # 1.1. 记录下labels
            self.labels = [str_util.to_str(la) for la in self.label_indexer_model.labels]

            # 1.2. 新生成编码后的列{label}_indexed，并删除{label}列
            X = self.label_indexer_model.transform(X).drop(label_col)

            # 1.3. {label}_indexed列重命名为 {label}列
            X = X.withColumnRenamed(self.get_indexed_label_col(), label_col)
        # fit 之后需要把string类型进行还原
        return X

    def get_indexed_label_col(self):
        # 约定编码后的label列，该列最终会被重命名。
        return self.get_label_col() + "_indexed"

    def get_estimator(self):
        return self.tuning_estimator.getEstimator()

    def fit_model(self, X, y=None, options=None, **kwargs):
        X = self.do_string_indexer_on_label(X)
        # 训练模型
        self.tuning_estimator = self.fit_input_model(self.tuning_estimator, X, y, options, **kwargs)
        return self.tuning_estimator.bestModel

    def get_params(self):
        """解析SKLearn模型的参数，从训练后的cv中解析出最优参数。
        Returns:
        """
        from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel
        if isinstance(self.tuning_estimator, CrossValidatorModel):
            metrics = "self.tuning_estimator.avgMetrics"
        elif isinstance(self.tuning_estimator, TrainValidationSplitModel):
            metrics = "self.tuning_estimator.validationMetrics"
        else:
            return None

        parameters = [({key.name: paramValue for key, paramValue in params.items()}, metric)
                      for params, metric in zip(self.tuning_estimator.getEstimatorParamMaps(), eval(metrics))]
        best_params = sorted(parameters, key=lambda x: x[1], reverse=True)[0][0]

        if validate_util.is_non_empty_list(best_params):
            return [Param(k, None, str(v)) for k, v in best_params.items()]
        else:
            return None
