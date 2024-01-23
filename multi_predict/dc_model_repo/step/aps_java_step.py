import abc

import pandas as pd
import six
import os

from dc_model_repo.base import FrameworkType, ModelFileFormatType
from dc_model_repo.base.data_sampler import PandasDataFrameDataSampler
from dc_model_repo.step.base import DCStep
from dc_model_repo.util import json_util


@six.add_metaclass(abc.ABCMeta)
class JavaObjectProxyDCStep(DCStep):
    """将特定Java对象包装成DCStep。
    """

    JVM_PORT = 25339

    def __init__(self, model_params, input_cols, algorithm_name):
        super(JavaObjectProxyDCStep, self).__init__(operator=None, framework=FrameworkType.APS,
                                                    model_format=ModelFileFormatType.PKL, input_cols=input_cols)
        self._fitted = True  # 这类对象仅用于上线，不需要训练。
        self.algorithm_name = algorithm_name
        self.model_params = model_params  # 模型训练参数
        self.jvm = self.get_jvm_connection(0, 3)  # 建立 py4j jvm 连接
        self.input_type = PandasDataFrameDataSampler.get_data_type()
        # 设置上线时候使用的参数
        self.extra = {
            "transformerParams": model_params
        }

    def get_jvm_connection(self, retry_times, max_times):
        self.ensure_start_jvm()  # 启动JVM
        from dc_model_repo.base.mr_log import logger
        from py4j.java_gateway import JavaGateway, GatewayParameters
        try:
            # 尝试连接JVM进程，超时时间5s
            gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.JVM_PORT, read_timeout=5))
            return gateway.jvm
        except Exception as e:
            if retry_times >= max_times:
                logger.error("连接JVM失败")
                logger.error(e)
                raise e  # 最后一次失败原因
            else:
                return self.get_jvm_connection(retry_times + 1, max_times)

    def exec_command(self, command):
        p = None
        try:
            p = os.popen(command, 'r')
            return p.read()
        except Exception as e:
            raise e
        finally:
            if p is not None:
                p.close()

    def find_java_sdk_jvm_pid(self):
        """检查java sdk jvm 进程id
        Returns:进程id，如果进程不存在返回 None
        """
        for line in self.exec_command("jps").split("\n"):
            data_tuple = line.split(" ")
            if len(data_tuple) > 1:
                if data_tuple[1] == "EntryPoint":
                    return int(data_tuple[0])
        # 2. 没有查找找存活的进程
        return None

    def ensure_start_jvm(self):
        """确保Java SDK JVM正常运行，如无法启动，则报错手动解决。`mrsdk-lib.jar`需要设置到classpath中。
        Returns:
        """
        from dc_model_repo.base.mr_log import logger
        # 1. 查找进程
        java_sdk_jvm_pid = self.find_java_sdk_jvm_pid()
        if java_sdk_jvm_pid is None:
            command = "javaaps -DserverPort=%s com.datacanvas.py4j.EntryPoint" % (self.JVM_PORT)
            logger.info("启动Java SDK JVM 进程，命令=%s" % (command))
            os.popen(command)
            import time
            time.sleep(1)
            # 1.1. 检查进程是否启动成功
            java_sdk_jvm_pid = self.find_java_sdk_jvm_pid()
            if java_sdk_jvm_pid is not None:
                logger.info("Java SDK JVM 进程启动成功, pid=%s" % java_sdk_jvm_pid)
            else:
                raise Exception("Java SDK JVM 进程启动失败，请检查mrsdk-lib.jar是否在classpath中。")
        else:
            logger.info("检查到Java SDK JVM 进程已经存在, pid=%s" % java_sdk_jvm_pid)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        # from py4j.java_gateway import java_import
        # java_import(self.jvm, 'java.util.*')
        self.jvm = self.get_jvm_connection(0, 3)  # 建立 py4j jvm 连接

        mapper = self.jvm.com.fasterxml.jackson.databind.ObjectMapper()
        java_model_params = mapper.readValue(json_util.to_json_str(self.model_params), self.jvm.java.util.Map._java_lang_class)
        self.model = self.deserialize_java_model(java_model_params)
        logger.info("成功加载Java模型: %s" % str(self.model))

    @abc.abstractmethod
    def deserialize_java_model(self, java_model_params):
        pass

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        # Spark模型无法序列化到pkl中
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["model", "jvm"])
        fs.write_bytes(step_path, obj_bytes)

    def transform(self, X, **kwargs):
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
        pd_data = {}
        for col in columns:
            pd_data[col] = []

        # 3. 合并预测结果，生成pandas 格式dict(要求所有dict里面的key一样)
        for prediction_map in result_list:
            for col in columns:
                pd_data.get(col).append(prediction_map.get(col))

        result_df = pd.DataFrame(data=pd_data, columns=columns)

        return result_df

    def get_params(self):
        # 仅用来预测，不需要params
        pass

    def persist_model(self, fs, destination):
        # 所有的信息已经序列化进meta.json
        pass


class JavaMissingTreatmentDCStep(JavaObjectProxyDCStep):

    def __init__(self, model_params):
        input_cols = model_params.get('inputCols')
        super(JavaMissingTreatmentDCStep, self).__init__(model_params, input_cols, 'MissingTreatment')

    def deserialize_java_model(self, java_model_params):
        instance = self.jvm.com.datacanvas.aps.evaluator.pipeline.transformer.MissingTreatment.load(java_model_params)
        return instance


class JavaDatetimeTransformerDCStep(JavaObjectProxyDCStep):

    def __init__(self, model_params):
        input_cols = [model_params.get('inputCol')]
        super(JavaDatetimeTransformerDCStep, self).__init__(model_params, input_cols, 'DatetimeTransformer')

    def deserialize_java_model(self, java_model_params):
        instance = self.jvm.com.datacanvas.aps.evaluator.pipeline.transformer.DatetimeTransformer.load(java_model_params)
        return instance
