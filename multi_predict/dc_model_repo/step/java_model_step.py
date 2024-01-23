import datetime
import numpy as np
from dc_model_repo.step.base import BaseEstimator
from py4j.java_gateway import JavaGateway, GatewayParameters
from py4j.protocol import Py4JNetworkError
import os
import time
from dc_model_repo.base.mr_log import logger
from collections import OrderedDict
import pandas as pd
import threading


class StartJavaThread(threading.Thread):
    def __init__(self, port=25339, timeout=120, jar_path=None, debug_log=False):
        threading.Thread.__init__(self)
        self.port = port
        self.timeout = timeout
        self.jar_path = jar_path
        self.debug_log = debug_log

    def run(self):
        if self.jar_path is None:
            command = "nohup javaaps -Dlogging.level.root={} -DserverPort={} com.datacanvas.py4j.EntryPoint &".format(
                "DEBUG" if self.debug_log else "INFO",
                self.port)
        else:
            command = "javaaps -Dlogging.level.root={} -DserverPort={} -cp {} com.datacanvas.py4j.EntryPoint".format(
                "DEBUG" if self.debug_log else "INFO",
                self.port,
                self.jar_path)
        logger.info("执行启动Java模型服务命令：{}".format(command))

        os.system(command)


def prepare_jvm(port=25339, timeout=120, jar_path=None, debug_log=False):

    StartJavaThread(port, timeout, jar_path, debug_log=debug_log).start()

    t0 = time.time()

    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))

    while True:
        try:
            if time.time() - t0 > timeout:
                raise Exception("无法在{}秒内启动Java模型服务，请排查问题".format(timeout))
            signal = gateway.jvm.java.lang.String("已成功连接JVM!")
            logger.info(signal)
            return gateway
        except Py4JNetworkError as e:
            logger.info("Java模型服务还没准备好. 再等待{}秒！参考信息：{}, cause: {} ".format(round(timeout + t0 - time.time()), str(e.args), str(e.cause)))
            time.sleep(1)


class FakeSparkStep:
    def __init__(self, input_type):
        self.input_type = input_type


class GeneratedJavaEstimator(BaseEstimator):

    def __init__(self, steps, model_path, jvm_port=None, jar_path=None, debug_log=False, **kwargs):
        """
        Args:
            steps: 要调用java端模型的连续step，类型为OrderedDict，key为step名称，value为加载的step对象
            model_path: DCPipeline模型路径
            jvm_port: 启动py4j服务的端口号
            jar_path: mrsdk的jar包路径。如果这个jar包已经在class_path上时候：不需要再设置。
            debug_log: 为True时通知java端打印debug级别日志，为False，打印info级别日志
            **kwargs: 备用
        """
        super(GeneratedJavaEstimator, self).__init__(**kwargs)
        assert isinstance(steps, OrderedDict)
        self.steps = steps
        self.model_path = model_path
        self.model_absolute_path = os.path.abspath(model_path)
        self.jar_path = jar_path
        self.jvm_port = 25339 if jvm_port is None else jvm_port
        self.gateway = None
        self.java_model = None
        self.debug_log = debug_log

    def prepare(self, step_path, **kwargs):
        """启动jvm，创建好java模型对象

        Returns: self
        """
        self.gateway = prepare_jvm(self.jvm_port, jar_path=self.jar_path, debug_log=self.debug_log)

        steps = [c for c in self.steps]
        string_class = self.gateway.jvm.String
        steps_array = self.gateway.new_array(string_class, len(steps))
        for i, s in enumerate(steps):
            steps_array[i] = s

        self.java_model = self.gateway.jvm.com.datacanvas.aps.evaluator.pipeline.DefaultPipeline("model", self.model_path, steps_array)

        return self

    def persist(self, destination=None, fs_type=None, persist_sample_data=False, persist_explanation=True, **kwargs):
        for step_name, step in self.steps.items():
            logger.info("Persist model: {}".format(step_name))
            step.persist(destination=destination, fs_type=fs_type, persist_sample_data=persist_sample_data, persist_explanation=persist_explanation, **kwargs)

    def predict(self, X, preserve_origin_cols=False, shutdown_java=True, **kwargs):
        """
        预测
        Args:
            X: 输入数据，pandas DataFrame类型
            preserve_origin_cols: 是否保留原有列，默认保留
            shutdown_java: 预测后关闭java端py4j服务
            **kwargs:

        Returns:
            预测结果
        """
        assert isinstance(X, pd.DataFrame), "当前构造的java模型输入只支持pandas DaraFrame格式。传入的却为：{}".format(str(type(X)))
        row_cnt = X.shape[0]
        batch_size = 1000
        quotient = row_cnt // batch_size
        remainder = row_cnt % batch_size
        split_points = [i*batch_size for i in range(quotient)]
        if remainder>0:
            split_points.append(batch_size*quotient)
        logger.info("当前数据需要分成{}次进行预测...".format(len(split_points)))
        result_dfs = []
        for i, (l, r) in enumerate(zip(split_points, split_points[1:]+[row_cnt])):
            logger.info("执行第{}次预测... [{}, {})".format(i+1, l, r))
            result_dfs.append(self._predict(X.iloc[l:r, :], preserve_origin_cols, shutdown_java, **kwargs))
        logger.info("合并预测结果...")
        result_df = pd.concat(result_dfs, axis=0)
        logger.info("预测结束.")
        return result_df

    def _predict(self, X, preserve_origin_cols=False, shutdown_java=True, **kwargs):
        """
        预测
        Args:
            X: 输入数据，pandas DataFrame类型
            preserve_origin_cols: 是否保留原有列，默认保留
            shutdown_java: 预测后关闭java端py4j服务
            **kwargs:

        Returns:
            预测结果
        """
        assert isinstance(X, pd.DataFrame), "当前构造的java模型输入只支持pandas DaraFrame格式。传入的却为：{}".format(str(type(X)))
        inputs = self.gateway.jvm.java.util.ArrayList()
        for row in X.iterrows():
            param_map = self.gateway.jvm.java.util.HashMap()
            for k, v in row[1].iteritems():
                if type(v) in [np.datetime64, pd.Timestamp, datetime.date]:
                    v = str(v)
                param_map.put(k, v)
            inputs.add(param_map)
        result = self.java_model.transform(inputs)
        cols = list(result[0].keySet())
        temp_dict = {c: [] for c in cols}
        for row in result:
            for c in cols:
                temp_dict[c].append(row[c])
        result_df = pd.DataFrame(temp_dict)
        if preserve_origin_cols is None or preserve_origin_cols:
            result_df = pd.concat([X, result_df], axis=1)

        return result_df

    def persist_model(self, fs, destination):
        pass

    def get_params(self):
        pass
