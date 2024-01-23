# -*- encoding: utf-8 -*-

import os
import abc
import time

import six

from dc_model_repo.base import StepType, FrameworkType, ChartData, Field, Output, TrainInfo
from dc_model_repo.base.mr_log import logger
from dc_model_repo.step.base import ModelWrapperDCStep, BaseEstimator
from dc_model_repo.util import validate_util, operator_output_util


@six.add_metaclass(abc.ABCMeta)
class KerasDCStep(ModelWrapperDCStep):
    def __init__(self, origin_model, model, kind, input_cols, algorithm_name, extension, **kwargs):
        super(KerasDCStep, self).__init__(kind=kind, framework=FrameworkType.Keras, model_format='h5',
                                          input_cols=input_cols, algorithm_name=algorithm_name,
                                          extension=extension, **kwargs)
        self.model = model
        self.origin_model = origin_model
        self.model_path = 'data/model.h5'

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["model", "origin_model", "callbacks", "sess",
                                                                    "sample_data", "target_sample_data"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination))
        model_path = model_path.replace("\\", "/")  # windows系统转换
        if not fs.exists(model_path):  # 模型解刨转换时需要创建model路径
            fs.make_dirs(model_path)
        h5_path = "%s/model.h5" % model_path
        self.model.save(h5_path)
        logger.info("keras model save to: %s" % h5_path)

        explanation = [ChartData('keras', 'netron', None, {"path": "data/model.h5"})]
        self.explanation = explanation

    def prepare(self, step_path, **kwargs):
        model_path = "%s/data/" % step_path
        logger.info("Prepare to load keras model at: %s." % model_path)
        t1 = time.time()

        import tensorflow as tf
        import tensorflow.python.keras.backend as K
        sess = K.get_session()
        graph = K.get_graph()

        with sess.as_default():
            with graph.as_default():
                self.model = tf.keras.models.load_model("%smodel.h5" % model_path)

        self.sess = sess
        self.graph = graph

        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("Load keras model: [%s] ,\ntook %s(s)." % (str(self.model), took))

    def get_params(self):
        """
        解析keras模型的参数
        :return:
        """
        from dc_model_repo.base import Param
        params = []
        # 增加输入列
        params.append(Param('input_cols', None, self.input_cols))
        return params

    def get_data_sampler(self, data):
        from dc_model_repo.base.data_sampler import ArrayDataSampler
        return ArrayDataSampler(data)

    def set_taininfo(self, x, elapsed):
        if isinstance(self, BaseEstimator):
            train_set_rows = None
            train_set_cols = None
            import pandas as pd
            if hasattr(x, 'shape'):
                train_set_rows = x.shape[0]
                train_set_cols = x.shape[1]

            elif isinstance(x, pd.DataFrame):
                train_set_rows = x.count()
                # fixme: spark 的序列化不准，封装成向量了。
                train_set_cols = len(x.columns)
            self.train_info = TrainInfo(train_set_rows, train_set_cols, elapsed)

    def get_estimator(self):
        """获取到评估器，可以从CV中提取。
        Returns:
        """
        return self.model

    def update_model(self):
        """更新模型。
        Returns:
        """
        pass

    def make_shape(self, shape):
        if shape is None:
            return [None]

        return list(shape)

    def fit_model(self, X, y=None, options=None, **kwargs):
        # 训练模型
        self.fit_input_model(self.model, X, y, **kwargs)
        return self.model

    def fit_input_model(self, input_model, X, y=None, batch_size=None,
                        epochs=1,
                        verbose=1,
                        callbacks=None,
                        validation_split=0.,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None, **kwargs):
        input_model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=verbose,
                        validation_split=validation_split, callbacks=callbacks,
                        validation_data=validation_data, class_weight=class_weight,
                        sample_weight=sample_weight, initial_epoch=initial_epoch,
                        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, **kwargs)
        return input_model

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        return self.get_data_sampler(y)

    def predict_result(self, X, predictions, batch_size=None,
                       verbose=0,
                       steps=None, calc_max_proba=True, calc_all_proba=False):

        import numpy as np

        if not isinstance(X, list):
            X = [X]
            predictions = [predictions]

        outputs = [o.name for o in self.outputs]
        result_dict = {}
        for i in range(0, len(outputs)):
            tmp_arr = []
            for j in range(len(predictions)):
                pred = [predictions[j]] if isinstance(predictions[j], np.ndarray) else predictions[j]
                tmp_arr.append(pred[i])
            result_dict.setdefault(outputs[i], tmp_arr)

        if hasattr(self.model, 'predict_proba'):
            if calc_max_proba or calc_all_proba:
                with self.sess.as_default():
                    with self.graph.as_default():
                        proba = self.model.predict_proba(X, batch_size=batch_size, verbose=verbose)
                proba_max = np.amax(proba, axis=1)
                max_prob_field = self.outputs[0].max_prob_field
                if max_prob_field is not None:
                    result_dict.setdefault(self.outputs[0].max_prob_field.name, proba_max)
                else:
                    result_dict.setdefault("%s_%s" % (outputs[0], "max_prob_field"), proba_max)

        return result_dict

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

    def get_targets(self, x, y=None, options=None, **kwargs):
        target_name = self.target_cols[0]
        output_field_type = self.get_as_pd_data_type(y)
        return [Field(target_name, output_field_type)]

    def get_outputs(self, x, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        # 1. 解析输出列的名称和类型
        output_field_type = self.get_as_pd_data_type(y)
        output_field_name = self.output_cols[0]  # 用户设置的输出列名称

        # 2. 如果有概率生成概率列
        output_field_raw_prob = None
        output_field_max_prob = None
        if hasattr(self.model, 'predict_proba'):
            # 1.1. 生成原始概率列
            output_field_raw_prob = operator_output_util.get_raw_prob_field(output_field_name, 'float64')

            # 1.2. 生成最大概率列
            output_field_max_prob = operator_output_util.get_max_prob_field(output_field_name, 'float64')

        # 2. 训练后检测class的个数并设置target
        output_fields_label_prob = None
        if hasattr(self.model, 'classes_'):
            classes_ = self.model.classes_
            if validate_util.is_non_empty_list(classes_):
                self.labels = classes_
                output_fields_label_prob = operator_output_util.get_label_prob_fields(output_field_name, classes_,
                                                                                      'float64')
                # label_prob_fields = [Feature('%s_proba_%s' % (target_name, c), 'float64') for c in classes_]
                #
                if output_field_raw_prob is not None:
                    output_field_raw_prob.shape = [1, len(classes_)]
                else:
                    logger.warning("模型:%s有分类的label:%s， 但是概率为空" % (str(classes_), str(self.model)))
            else:
                logger.warning("模型: %s有分类的label， classes为空。" % str(self.model))
        # max_prob_field=None, raw_prob_field=None, label_prob_fields
        output = Output(output_field_name, output_field_type, None, output_field_max_prob, output_field_raw_prob,
                        output_fields_label_prob)
        return [output]


class KerasDCCustomerEstimator(KerasDCStep, BaseEstimator):

    def __init__(self, model, input_cols, target_cols=None, algorithm_name=None, extension=None):
        # 1. 检查target_cols
        if target_cols is None or len(target_cols) == 0:
            target_cols = ["prediction"]

        self.target_cols = target_cols
        self.model = model  # 记录cv

        # 2. 调用父类构造方法
        super(KerasDCCustomerEstimator, self).__init__(origin_model=model, model=model, kind=StepType.Estimator,
                                                       input_cols=input_cols, algorithm_name=algorithm_name,
                                                       extension=extension)

        self.output_cols = self.target_cols

    def predict(self, X, batch_size=None,
                verbose=0,
                steps=None, calc_max_proba=True, calc_all_proba=False, **kwargs):

        with self.sess.as_default():
            with self.graph.as_default():
                predictions = self.model.predict(X, batch_size=batch_size, verbose=verbose, steps=steps)

        return self.predict_result(X, predictions, batch_size, verbose, steps, calc_max_proba, calc_all_proba)

    def transform(self, df, **kwargs):
        return self.predict(df, **kwargs)


class KerasDCEstimator(KerasDCStep, BaseEstimator):

    def __init__(self, model, algorithm_name=None, **kwargs):
        # 2. 调用父类构造方法
        super(KerasDCEstimator, self).__init__(origin_model=model, model=model, kind=StepType.Estimator,
                                               input_cols=None, algorithm_name=algorithm_name, extension=None,
                                               **kwargs)

    def predict(self, X, batch_size=None, verbose=0, steps=None,
                calc_max_proba=True, calc_all_proba=False):
        if isinstance(X, dict):
            data = []
            for input in self.input_features:
                data.append(X[input.name])
            X = data

        with self.sess.as_default():
            with self.graph.as_default():
                predictions = self.model.predict(X, batch_size=batch_size, verbose=verbose, steps=steps)

        return self.predict_result(X, predictions, batch_size, verbose, steps, calc_max_proba, calc_all_proba)

    def transform(self, df, **kwargs):
        return self.predict(df, **kwargs)

    def fit_input_model(self, input_model, X, y=None, **kwargs):
        return input_model

    def fit(self, X_train, y_train=None, input_cols=None, target_cols=None, output_cols=None,
            algorithm_name=None, options=None, output_type=None, extension=None, **kwargs):
        if target_cols is None or len(target_cols) == 0:
            target_cols = ["prediction"]

        if input_cols is None:
            input_cols = []

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
        self.output_cols = output_cols
        if output_cols is None:
            self.output_cols = target_cols

        if algorithm_name is not None:
            self.algorithm_name = algorithm_name

        if output_type is not None:
            self.output_type = output_type

        super(KerasDCEstimator, self).fit(X_train, y_train)
