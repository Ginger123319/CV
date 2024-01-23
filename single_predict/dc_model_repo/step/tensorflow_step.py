# -*- encoding: utf-8 -*-

import abc
import os
import time

import pandas as pd
import six
import tensorflow as tf

from dc_model_repo.base import Field, ChartData, Output
from dc_model_repo.base import StepType, FrameworkType, ModelFileFormatType
from dc_model_repo.base.mr_log import logger
from dc_model_repo.step.base import ModelWrapperDCStep, BaseEstimator


@six.add_metaclass(abc.ABCMeta)
class TensorflowDCStep(ModelWrapperDCStep):
    def __init__(self, origin_model, model, kind, input_cols, algorithm_name, extension, **kwargs):
        super(TensorflowDCStep, self).__init__(kind=kind,
                                               framework=FrameworkType.TensorFlow,
                                               model_format=ModelFileFormatType.CKPT,
                                               input_cols=input_cols, algorithm_name=algorithm_name,
                                               extension=extension, **kwargs)
        self.model_path = 'data/model'
        self.origin_model = origin_model
        self.model = None
        self.feature_importance = None
        self.tensor_dtypes = {0: "invalid", 1: "float", 2: "double", 3: "int32", 4: "uint8", 5: "int16", 6: "int8",
                              7: "string", 8: "complex64", 9: "int64", 10: "bool"}
        self.limit = 1

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize_with_ignore_variables(self,
                                                             ["origin_model", "session", "input_nodes", "output_nodes",
                                                              "sample_data"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model')
        model_path = model_path.replace("\\", "/")  # windows系统转换
        # if not fs.exists(model_path):  # 模型解刨转换时需要创建model路径
        #     fs.make_dirs(model_path)
        # model_path = "data/model"

        import uuid
        temp_path = str(uuid.uuid4())

        if self.saver_type == ModelFileFormatType.CKPT:
            saver = tf.train.Saver()
            saver.save(self.session, ("%s/model" % temp_path))

            # fs.copy(temp_path, model_path)
            model_tmp_path = os.path.join(self.serialize_data_path(destination), temp_path)

            import shutil
            shutil.move(temp_path, model_tmp_path)
            os.rename(model_tmp_path, model_path)

            explanation = [ChartData('tensorboard', 'tensorflow', None, {"path": "data/model"})]
            self.explanation = explanation
        else:
            if self.original_model_path is not None:
                # 不重新序列化，直接copy过来
                import shutil
                shutil.copy(self.original_model_path, '%s.pb' % model_path)
            else:
                from tensorflow.python.framework import graph_util

                constant_graph = graph_util.convert_variables_to_constants(self.session, self.session.graph_def, self.output_cols)

                # 写入序列化的 PB 文件
                with tf.gfile.FastGFile('%s.pb' % model_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

            explanation = [ChartData('tensorboard', 'netron', None, {"path": "data/model.pb"})]
            self.explanation = explanation

    def prepare(self, step_path, **kwargs):
        model_path = "%s/data/model/" % step_path
        logger.info("Prepare to load tensorflow model at: %s" % model_path)
        t1 = time.time()

        tf.reset_default_graph()
        self.session = tf.Session()

        if self.saver_type == ModelFileFormatType.PB:
            from tensorflow.python.platform import gfile
            with gfile.FastGFile(('%s/data/model.pb' % step_path), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.session.graph.as_default()
                tf.import_graph_def(graph_def, name='')  # 导入计算图

            self.session.run(tf.initialize_all_variables())

            # for x in self.input_features:
            #     x.name = "import/%s" % x.name
            #
            # for x in self.outputs:
            #     x.name = "import/%s" % x.name
        else:
            self.session.run(tf.initialize_all_variables())
            saver = tf.train.import_meta_graph("%smodel.meta" % model_path)
            saver.restore(self.session, tf.train.latest_checkpoint(model_path))

        self.model = tf.get_default_graph()

        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("Load tensorflow model ,\ntook %s(s)." % (took))

    def get_params(self):
        """
        解析Tensorflow模型的参数
        :return:
        """
        from dc_model_repo.base import Param
        params = []
        for k in self.input_cols:
            params.append(Param(k, None, None))

        # 增加输入列
        params.append(Param('input_cols', None, self.input_cols))
        return params

    def make_shape(self, shape):
        if shape is None or shape.dims is None:
            return [None]

        return shape.as_list()

    def parse_feature(self, node):
        values = node.attr
        shape = []

        if values.__contains__("shape"):
            shape_dim = values["shape"].shape.dim
            for item in range(len(shape_dim)):
                if shape_dim[item].size <= 0:
                    shape.append(1)
                else:
                    shape.append(shape_dim[item].size)

        if len(shape) == 0:
            shape = None

        if values.__contains__("T"):
            type = self.tensor_dtypes[values["T"].type]
            if type is None:
                type = values["T"].type
            return Field(node.name, type, shape)

        if values.__contains__("SrcT"):
            type = self.tensor_dtypes[values["SrcT"].type]
            if type is None:
                type = values["SrcT"].type
            return Field(node.name, type, shape)

        type = self.tensor_dtypes[values["dtype"].type]
        if type is None:
            type = values["dtype"].type

        return Field(node.name, type, shape)

    def is_const(self, op_name):
        """Return True if node is a constant."""
        return op_name in ["Const", "ConstV2", "Constant"]

    def is_graph_input(self, op_name):
        return op_name in ["Placeholder", "PlaceholderV2"]  # "PlaceholderWithDefault",

    def is_graph_output(self, op_name):
        return op_name in ["Softmax", "Sigmoid", "Tanh", "Relu", "Softmax_v2", "Log_softmax", "Log_softmax_v2",
                           "Softmax_cross_entropy_with_logits", "Softmax_cross_entropy_with_logits_v2",
                           "Sparse_softmax_cross_entropy_with_logits", "Sparse_softmax_cross_entropy_with_logits_v2",
                           "Avg_pool", "Avg_pool_v2",
                           "Avg_pool1d", "Avg_pool2d", "Avg_pool3d",
                           "Max_pool", "Max_pool_v2",
                           "Max_pool1d", "Max_pool2d", "Max_pool3d", "Max_pool_with_argmax", "Max_pool_with_argmax_v1",
                           "Dropout", "Dropout_v2", "Top_k", "Fractional_max_pool", "Fractional_avg_pool", "Erosion2d",
                           "In_top_k",
                           "Quantized_avg_pool", "Quantized_conv2d", "Quantized_relu_x", "Quantized_max_pool"
                           ]

    def get_tensor_nodes(self):
        nodes = self.session.graph.as_graph_def().node
        self.input_nodes = []
        self.output_nodes = []

        for node in nodes:
            if self.is_const(node.op) and node.name in self.input_cols:
                self.input_nodes.append(node)

            if self.is_graph_input(node.op) and node.name in self.input_cols:
                self.input_nodes.append(node)

            if self.is_graph_output(node.op) and node.name in self.target_cols:
                self.output_nodes.append(node)

    def fit_model(self, X, y=None, options=None, **kwargs):
        return self.model

    def get_as_pd_data_type(self, data):
        """将输出的numpy数据转换成dataframe，然后获取其类型。
        Args:
            data: numpy数组，并且只有一列。
        Returns:
        """
        import pandas as pd
        import numpy as np

        if isinstance(data, np.ndarray):
            return data.dtype.name
        elif isinstance(data, pd.DataFrame):
            return list(data.dtypes.to_dict().values())[0].name

        return ""

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        return self.get_data_sampler(y)

    def get_targets(self, x, y=None, options=None, **kwargs):
        targets = []
        for node in self.output_nodes:
            targets.append(self.parse_feature(node))

        self.target = targets
        return self.target

    def get_input_features(self, x, y=None, options=None, **kwargs):
        input_features = []
        for node in self.input_nodes:
            input_features.append(self.parse_feature(node))

        self.input_features = input_features
        return self.input_features

    def get_outputs(self, x, y=None, options=None, **kwargs):
        outputs = []
        for node in self.output_nodes:
            node_feature = self.parse_feature(node)
            outputs.append(Output(node_feature.name, node_feature.type, node_feature.shape, None, None, None))

        return outputs


class TensorflowDCEstimator(TensorflowDCStep, BaseEstimator):

    def __init__(self, session, original_model_path=None, algorithm_name=None, saver_type=ModelFileFormatType.CKPT, **kwargs):
        # 2. 调用父类构造方法
        super(TensorflowDCEstimator, self).__init__(origin_model=session, model=session, kind=StepType.Estimator,
                                                    input_cols=None, algorithm_name=algorithm_name,
                                                    extension=None, **kwargs)

        self.session = session
        self.saver_type = saver_type
        self.original_model_path = original_model_path

    def predict(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        import numpy as np
        inputs = [i.name for i in self.input_features]
        outputs = [o.name for o in self.outputs]

        if isinstance(X, list):
            input_feed = {self.model.get_tensor_by_name("%s:0" % inputs[i]): X[i] for i in range(len(inputs))}
        elif isinstance(X, np.ndarray):
            input_feed = {self.model.get_tensor_by_name("%s:0" % inputs[i]): X for i in range(len(inputs))}
        elif isinstance(X, pd.DataFrame):
            input_feed = {self.model.get_tensor_by_name("%s:0" % inputs[i]): X[inputs[i]][0] for i in
                          range(len(inputs))}
        elif isinstance(X, dict):
            input_feed = {self.model.get_tensor_by_name("%s:0" % inputs[i]): X[inputs[i]] for i in range(len(inputs))}
        else:
            raise Exception('not support data type {}'.format(type(X)))

        predict_y = []
        for k in outputs:
            predict_y.append(self.model.get_tensor_by_name("%s:0" % k))

        predictions = self.session.run(predict_y, feed_dict=input_feed)

        result_dict = {}
        for i in range(0, len(outputs)):
            result_dict.setdefault(outputs[i], predictions[i])

        return result_dict

    def transform(self, X):
        return self.predict(X)

    def fit(self, X_train, y_train=None, input_cols=[], target_cols=[],
            algorithm_name=None, options=None, extension=None, **kwargs):

        if target_cols is None or len(target_cols) == 0:
            target_cols = ["prediction"]

        if isinstance(X_train, dict):
            data_x = []
            data_y = []
            for name in input_cols:
                data_x.append(X_train[name])

            for name in target_cols:
                data_y.append(X_train[name])

            X_train = data_x
            y_train = data_y

        self.input_cols = input_cols
        self.target_cols = target_cols
        self.output_cols = target_cols

        if algorithm_name is not None:
            self.algorithm_name = algorithm_name

        # 获取tensor_nodes
        self.get_tensor_nodes()

        super(TensorflowDCEstimator, self).fit(X_train, y_train)

        self.get_input_features(X_train, y_train)
