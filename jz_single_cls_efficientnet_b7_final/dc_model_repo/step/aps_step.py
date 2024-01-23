# -*- encoding: utf-8 -*-
from os import path as P

import os
import pandas as pd

from dc_model_repo.base import FrameworkType, ModelFileFormatType
from dc_model_repo.step.base import DCStep, BaseEstimator, BaseTransformer
from dc_model_repo.step.sklearn_step import SKLearnLikePredictDCEstimator
from dc_model_repo.util import validate_util
from pyspark.sql import DataFrame as SparkDataFrame
from dc_model_repo.base.data_sampler import ArrayDataSampler, DictDataSampler
import os.path

class APS31CustomStep(SKLearnLikePredictDCEstimator):
    """自定义模型格式（龙湖）升级到APS3.2格式专用。"""

    def __init__(self, serving_model_dir):
        super(APS31CustomStep, self).__init__(None, input_cols=None, algorithm_name="ModelServing",
                                              framework=FrameworkType.APS, model_format=ModelFileFormatType.ZIP)
        self._fitted = True
        validate_util.require_str_non_empty(serving_model_dir, "serving_model_dir")
        if not os.path.exists(serving_model_dir):
            raise Exception("模型路径不存在: %s" % serving_model_dir)

        serving_file_path = os.path.join(serving_model_dir, 'model_serving.py')
        has_serving_file = False
        if not os.path.exists(serving_file_path):
            files = os.listdir(serving_model_dir)
            for f in files:
                if os.path.isdir(os.path.join(serving_model_dir, f)):
                    serving_file_path = os.path.join(serving_model_dir, f, 'model_serving.py')
                    if os.path.exists(serving_file_path):
                        serving_model_dir = os.path.join(serving_model_dir, f)
                        has_serving_file = True
                        break
        else:
            has_serving_file = True
        if not has_serving_file:
            raise Exception("serving文件不存在: %s" % serving_file_path)

        if not P.isdir(serving_model_dir):
            raise Exception("serving_model_dir=%s必须是目录。" % serving_model_dir)

        self.serving_model_dir = serving_model_dir
        BaseEstimator.__init__(self, None)

    def persist_model(self, fs, destination):
        """将自定义模型的数据复制到sourceCode目录中。

        Args:
            fs:
            destination:
        Returns:
        """
        # 1. 将自定义模型目录复制到data下的sourceCode目录，例如：/tmp/custom/model_serving.py => /tmp/uuid/steps/1_s/data/sourceCode/model_serving.py
        serialize_source_code_path = self.serialize_source_code_path(destination)
        # 1.1. 确保step/data/sourceCode目录不存在
        if P.exists(serialize_source_code_path):
            raise Exception("Step 的源码目录不为空，无法将model_serving文件夹复制进去。")

        # 1.2. 复制目录并改名 (serving_model_dir是目录，已经校验过)
        fs.copy(self.serving_model_dir, serialize_source_code_path)

    def prepare(self, step_path, **kwargs):
        from model_serving import ModelServing
        self.model = ModelServing()

    def get_params(self):
        return None


class PipelineInitDCStep(DCStep):
    """处理DCPipeline初始化相关的信息采集。"""

    def __init__(self, label_col, **kwargs):
        super(PipelineInitDCStep, self).__init__(operator=None,
                                                 framework=FrameworkType.APS,
                                                 model_format=ModelFileFormatType.PKL,
                                                 input_cols=None,
                                                 algorithm_name='PipelineInitDCStep',
                                                 **kwargs)
        self.label_col = label_col  # 标签列名，当为Spark DataFrame 时候必须传
        if label_col is None:
            from dc_model_repo.base.mr_log import logger
            logger.warn("Pipeline初始化设置的label_col为None，将无法直接采集标签列样本")
        self.target_sample_data = None  # 除了 Estimator, 初始化模块也有标签列样本数据。

    def fit_prepare(self, X, y=None, options=None, **kwargs):
        """获取数据集的格式。

        Args:
            X: 训练数据集，如果是PySpark DF 需要包含label 的dafaframe。
            y: 输入为Pandas或Dict时需要传，Spark 不需要传。
            options(dict): 送给 :attr:`operator` 的 ``fit`` 方法的参数。
            **kwargs: 扩展字段。

        Returns: X
        """
        from dc_model_repo.base.mr_log import logger

        # 1. 读取输入数据的schema
        data_sampler = self.get_feature_sample_data(X)
        input_features = data_sampler.get_input_features()

        self.input_cols = None if input_features is None else [f.name for f in input_features if f.name!=self.label_col]

        # 2. 处理label列
        if isinstance(X, SparkDataFrame):
            # 2.1. 从Spark 的DataFrame中截取y
            if self.label_col is not None:
                y = X.select([self.label_col])
            else:
                logger.warning("Spark DataFrame没有设置 label_col，无法从当前数据集中获取标签列样本数据。")

        elif isinstance(X, pd.DataFrame):
            # 2.2. 如果X是pandas的，y同样使用pandas抽样方法
            if y is None:
                logger.warning("Pandas DataFrame 没有设置 参数y，无法获取标签列样本数据。")
            else:
                if not isinstance(y, pd.DataFrame):
                    y = pd.DataFrame(data=y)
        elif isinstance(data_sampler, ArrayDataSampler):
            if y is None:
                logger.warning("Array Data 没有设置 参数y，无法获取标签列样本数据。")
            pass
        elif DictDataSampler.is_compatible(X):
            if y is None and (self.label_col is not None):
                if self.label_col in X:
                    try:
                        y = pd.DataFrame(X.pop(self.label_col))
                    except Exception as e:
                        logger.error("Can't create dataframe with current value: {}".format(repr(e)))
                        y = None
            if y is None:
                logger.warning("当前输入为Dict格式数据，没有设置标签列")
        else:
            # 其他类型的数据当成数组处理
            pass

        # 3. 校验数据
        if y is not None:
            # 3.1. 读取标签列样本数据(同时支持Spark DataFrame、SKLearn DataFrame、Numpy)
            self.target_sample_data = self.get_data_sampler(y)
        else:
            logger.warning("输入的参数y为空，并且无法从lable_col中识别标签数据。")

        return X

    def get_feature_sample_data(self, X):
        if isinstance(X, SparkDataFrame):
            # Spark DataFrame 并且设置了label_col 删除 label
            if self.label_col is not None:
                return self.get_data_sampler(X.drop(self.label_col))

        return super(PipelineInitDCStep, self).get_feature_sample_data(X)

    def persist_model(self, fs, destination):
        pass

    def prepare(self, step_path, **kwargs):
        pass

    def get_params(self):
        pass


class PipelineSampleDataCollectDCStep(PipelineInitDCStep):
    """
    处理DCPipeline初始化相关的信息采集，V3.2使用：
        - 不支持采集标签样本数据
        - 不支持Spark string类型的标签列训练
        - 不管单机还是分布式传入X都不能带有y
    """

    def __init__(self, **kwargs):
        from dc_model_repo.base.mr_log import logger
        logger.warning("PipelineSampleDataCollectDCStep将要被废弃，请使用PipelineInitDCStep代替。")
        super(PipelineSampleDataCollectDCStep, self).__init__(label_col=None)


class PMMLDCStep(BaseEstimator):

    def __init__(self, pmml_path, input_cols=None, algorithm_name='PMMLDCStep'):
        super(PMMLDCStep, self).__init__(operator=None, framework=FrameworkType.APS,
                                         model_format=ModelFileFormatType.PMML,
                                         input_cols=input_cols)
        self._fitted = True
        self.algorithm_name = algorithm_name
        self.model_path = 'data/model.pmml'
        self.pmml_path = pmml_path
        self.jvm_port = 25339
        self.try_start_jvm_count = 1

        self.load(pmml_path)

    def predict_(self, X, **kwargs):
        x_columns = X.columns.values.tolist()
        result_list = []
        # 1. 遍历执行transform
        for index, row in X.iterrows():
            # 1.1. 组装参数
            row_map = self.jvm.java.util.HashMap()
            for column in X.columns:
                row_map.put(column, row[column])

            # 1.2. 预测
            prediction_map = dict(self.model.transform(row_map))
            result_list.append(prediction_map)

        # 2. 初始化pandas 数据
        columns = result_list[0].keys()  # 数据不能为空
        result_columns = []
        pd_data = {}
        for col in columns:
            col_pred = col.replace("_pred", "")
            if col_pred in x_columns:
                pd_data[col] = []
                result_columns.append(col)
            else:
                pd_data[col_pred] = []
                result_columns.append(col_pred)

        # 3. 合并预测结果，生成pandas 格式dict(要求所有dict里面的key一样)
        for prediction_map in result_list:
            for col in columns:
                col_pred = col.replace("_pred", "")
                if col_pred in x_columns:
                    pd_data.get(col).append(prediction_map.get(col))
                else:
                    pd_data.get(col_pred).append(prediction_map.get(col))

        result_df = pd.DataFrame(data=pd_data, columns=result_columns)
        for col in result_columns:
            X[col] = result_df[col]

        return X

    def predict(self, X, **kwargs):  # 替换测试
        data = []
        columns = []
        for index, row in X.iterrows():
            # 组装参数
            map = self.jvm.java.util.HashMap()
            for column in X.columns:
                map.put(column, row[column])
            # 进行预测
            prediction = self.model.transform(map)

            # 解析预测结果
            if index == 0:
                columns, columnsName = self.get_columns(prediction)

            row_data = []
            for name in columnsName:
                row_data.append(prediction.get(name))

            data.append(row_data)

        # 组成DF
        import pandas as pd
        df = pd.DataFrame(data=data, columns=columns)
        for col in columns:
            X[col] = df[col]

        return X

    def get_columns(self, prediction):
        from dc_model_repo.base import Output

        columnsName = []
        columns = []
        iterator = prediction.keySet().iterator()
        while iterator.hasNext():
            name = iterator.next()

            if self.outputs is not None:
                for o in range(len(self.outputs)):
                    output = self.outputs[o]
                    has_name = False
                    if output is not None and isinstance(output, Output):
                        if "_pred" in name:
                            columns.append(output.name)
                            columnsName.append(name)
                            has_name = True

                        if "_prob" in name and output.max_prob_field is not None:
                            columns.append(output.max_prob_field.name)
                            columnsName.append(name)
                            has_name = True

                        if has_name is False:
                            columns.append(output.name)
                            columnsName.append(name)

            else:
                columns.append(name.replace("_pred", ""))
                columnsName.append(name)
        return columns, columnsName

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        # Spark模型无法序列化到pkl中
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["model", "jvm", "app"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pmml')
        fs.copy(self.pmml_path, model_path)

    def transform(self, X):
        self.predict(X)

    def load(self, model_path):
        transformer = self.try_get_transformer(5)

        # # 读取文件
        f = open(model_path, "r")
        pmml_content = f.read()
        f.close()

        # 加载pmml
        self.model = transformer.loadFromPmmlString(pmml_content)

    def get_transformer(self):
        from py4j.java_gateway import JavaGateway, GatewayParameters
        # 初始化gateway
        gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.jvm_port))
        self.jvm = gateway.jvm
        transformer = self.jvm.com.datacanvas.aps.evaluator.pipeline.transformer.SinglePMMLDTransformer

        return transformer

    def try_get_transformer(self, try_count=1):
        try:
            transformer = self.get_transformer()
        except Exception as e:
            if not hasattr(self, 'try_start_jvm_count'):
                self.try_start_jvm_count = 1

            if self.try_start_jvm_count <= try_count:
                # 开启jvm
                self.start_jvm()
                self.try_start_jvm_count = self.try_start_jvm_count + 1
                transformer = self.try_get_transformer(try_count)

        return transformer

    def start_jvm(self):
        try:
            from dc_model_repo.base.mr_log import logger
            import platform
            import os

            # command = "java -DserverPort=%s -cp E:\\wyc\\project\\dc-sdk-mr-py\\jars\\mrsdk-lib.jar com.datacanvas.py4j.EntryPoint" % (self.jvm_port)
            command = "javaaps -DserverPort=%s com.datacanvas.py4j.EntryPoint" % (self.jvm_port)
            logger.info("尝试第%s次开启jvm: \n%s" % (self.try_start_jvm_count, command))
            os.popen(command)  # 开启子线程

            import time

            time.sleep(4)

        except Exception as e:
            pass

    def prepare(self, step_path, **kwargs):
        """从HDFS上加载Step。

        Args:
            step_path:
            **kwargs:

        Returns:

        """
        from dc_model_repo.base.mr_log import logger
        # 1. 使用Pipeline加载模型
        p_model = step_path + "/data/model.pmml"
        self.load(p_model)
        logger.info("成功加载Spark模型: %s" % str(self.model))

    def get_params(self):
        pass


class ProxyPMMLDCStep(BaseEstimator):

    def __init__(self, pmml_path=None, class_name=None, model_params=None, input_cols=None,
                 algorithm_name='ProxyPMMLDCStep'):
        super(ProxyPMMLDCStep, self).__init__(operator=None, framework=FrameworkType.APS,
                                              model_format=ModelFileFormatType.PMML,
                                              input_cols=input_cols)
        self._fitted = True
        self.algorithm_name = algorithm_name
        self.model_path = 'data/model.pmml'
        self.pmml_path = pmml_path
        self.class_name = class_name
        self.model_params = model_params
        self.jvm_port = 25339
        self.try_start_jvm_count = 1

        if pmml_path is not None:
            self.load(True, pmml_path)
        else:
            self.load(False)

    def predict(self, X, **kwargs):
        from py4j.java_gateway import java_import

        java_import(self.jvm, 'java.util.*')

        data = []
        columns = []
        for index, row in X.iterrows():
            # 组装参数
            map = self.jvm.HashMap()
            for column in X.columns:
                map.put(column, row[column])
            # 进行预测
            prediction = self.model.transform(map)

            # 解析预测结果
            if index == 0:
                columns, columnsName = self.get_columns(prediction)

            row_data = []
            for name in columnsName:
                row_data.append(prediction.get(name))

            data.append(row_data)
        # 组成DF
        import pandas as pd
        df = pd.DataFrame(data=data, columns=columns)

        return df

    def get_columns(self, prediction):
        from dc_model_repo.base import Output

        columnsName = []
        columns = []
        iterator = prediction.keySet().iterator()
        while iterator.hasNext():
            name = iterator.next()

            if self.outputs is not None:
                for o in range(len(self.outputs)):
                    output = self.outputs[o]
                    if output is not None and isinstance(output, Output):
                        if "_pred" in name:
                            columns.append(output.name)
                            columnsName.append(name)

                        if "_prob" in name and output.max_prob_field is not None:
                            columns.append(output.max_prob_field.name)
                            columnsName.append(name)
            else:
                columns.append(name.replace("_pred", ""))
                columnsName.append(name)
        return columns, columnsName

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        # Spark模型无法序列化到pkl中
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["model", "jvm", "app"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        if self.pmml_path is not None:
            model_path = os.path.join(self.serialize_data_path(destination), 'model.pmml')
            fs.copy(self.pmml_path, model_path)

    def transform(self, X):
        return self.predict(X)

    def read_file(self, file_path):
        # 读取文件
        f = open(file_path, "r")
        content = f.read()
        f.close()

        return content

    def load(self, is_pmml=False, pmml_path=""):
        transformer = self.try_get_transformer(5)

        if is_pmml:
            # 读取文件
            pmml_content = self.read_file(pmml_path)

            # 加载pmml
            self.model = transformer.loadFromPmmlString(pmml_content)
        else:
            from py4j.java_gateway import java_import
            import json

            java_import(self.jvm, 'java.util.*')

            data = json.dumps(self.model_params)

            mapper = self.jvm.com.fasterxml.jackson.databind.ObjectMapper()
            metaMap = mapper.readValue(data, self.jvm.Map._java_lang_class)

            # mapper = self.jvm.com.alibaba.fastjson.JSONObject
            # metaMap = mapper.parseObject(data, self.jvm.Map._java_lang_class)

            self.model = transformer.load(metaMap)

    def get_transformer(self):
        from py4j.java_gateway import JavaGateway, GatewayParameters
        # 初始化gateway
        gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.jvm_port))
        self.jvm = gateway.jvm

        if self.class_name == "dc_builtin_step.dc_pyspark.missing_treatment.MissingTreatment":
            transformer = self.jvm.com.datacanvas.aps.evaluator.pipeline.transformer.MissingTreatment
        elif self.class_name == "dc_builtin_step.dc_pyspark.datetime_transformer.DatetimeTransformer":
            transformer = self.jvm.com.datacanvas.aps.evaluator.pipeline.transformer.DatetimeTransformer
        else:
            transformer = self.jvm.com.datacanvas.aps.evaluator.util.PMMLDTransformer

        return transformer

    def try_get_transformer(self, try_count=1):
        try:
            transformer = self.get_transformer()
        except Exception as e:
            if not hasattr(self, 'try_start_jvm_count'):
                self.try_start_jvm_count = 1

            if self.try_start_jvm_count <= try_count:
                # 开启jvm
                self.start_jvm()
                self.try_start_jvm_count = self.try_start_jvm_count + 1
                transformer = self.try_get_transformer(try_count)

        return transformer

    def start_jvm(self):
        try:
            from dc_model_repo.base.mr_log import logger
            import platform
            import os

            # command = "java -DserverPort=%s -cp D:\\project\\python\\dc-sdk-mr-py\\jars\\mrsdk-lib.jar com.datacanvas.py4j.EntryPoint" % (
            #     self.jvm_port)
            command = "javaaps -DserverPort=%s com.datacanvas.py4j.EntryPoint" % (self.jvm_port)
            logger.info("尝试第%s次开启jvm: \n%s" % (self.try_start_jvm_count, command))
            os.popen(command)  # 开启子线程

            import time

            time.sleep(4)

        except Exception as e:
            pass

    def prepare(self, step_path, **kwargs):
        """
        从HDFS上加载Step。
        :param path:
        :param kwargs:
        :return:
        """
        from dc_model_repo.base.mr_log import logger
        from dc_model_repo.util import json_util

        # 1. 使用Pipeline加载模型
        step_meta_path = "%s/meta.json" % (step_path)
        meta_str = self.read_file(step_meta_path)
        meta_data = json_util.to_object(meta_str)
        className = meta_data["className"]
        if className in ["dc_builtin_step.dc_pyspark.missing_treatment.MissingTreatment",
                         "dc_builtin_step.dc_pyspark.datetime_transformer.DatetimeTransformer"]:
            self.load(False)
        else:
            pmml_path = "%s/data/model.pmml" % step_path
            self.load(True, pmml_path=pmml_path)

        logger.info("成功加载Spark模型: %s" % str(self.model))

    def get_params(self):
        pass


class FakeUnSerializableDCStep(BaseTransformer):
    """
    假冒无法反序列化的DCStep
    """

    def __init__(self, step_meta):
        super(FakeUnSerializableDCStep, self).__init__(operator=None,
                                                       framework=step_meta.framework,
                                                       model_format=step_meta.model_format,
                                                       input_cols=[f.name for f in step_meta.input_features],
                                                       algorithm_name=step_meta.algorithm_name)

        self.id = step_meta.id  # 需要使用之前step的id
        self.module_id = step_meta.module_id

        self.step_meta = step_meta
        # 设置训练参数
        self.params = step_meta.params
        # 设置扩展属性
        self.extension = step_meta.extension

    def persist_model(self, fs, destination):
        pass

    def get_params(self):
        pass

    def transform(self, X, **kwargs):
        raise Exception("当前DCStep无法执行transform，因为并没有反序列化模型。")

    def predict(self, X, **kwargs):
        self.transform(X, **kwargs)


class FakeUnSerializableDCEstimator(BaseEstimator):

    def __init__(self, step_meta):
        super(FakeUnSerializableDCEstimator, self).__init__(operator=None,
                                                            framework=step_meta.framework,
                                                            model_format=step_meta.model_format,
                                                            input_cols=[f.name for f in step_meta.input_features],
                                                            algorithm_name=step_meta.algorithm_name,
                                                            output_cols=[o.name for o in step_meta.outputs],
                                                            target_cols=[t.name for t in step_meta.target])

        self.id = step_meta.id  # 需要使用之前step的id
        self.module_id = step_meta.module_id

        self.step_meta = step_meta
        # 设置训练参数
        self.params = step_meta.params
        # 设置扩展属性
        self.extension = step_meta.extension

        # 1. 设置target
        self.target = step_meta.target

        # 2. 设置概率
        self.outputs = step_meta.outputs

        # 3. 设置算法名称
        self.algorithm_name = step_meta.algorithm_name

        # 4. 设置训练信息
        self.train_info = step_meta.train_info

    def persist_model(self, fs, destination):
        pass

    def get_params(self):
        pass

    def transform(self, X, **kwargs):
        raise Exception("当前DCStep无法执行transform，因为并没有反序列化模型。")

    def predict(self, X, **kwargs):
        self.transform(X, **kwargs)