# -*- encoding: utf-8 -*-

import abc
from dc_seldon.datacanvas.aps.proto.prediction_pb2 import Dict
from dc_model_repo.base.data_sampler import DictDataSampler
import os
from pickle import NONE
import numpy as np
import six
import time
import tensorflow as tf
from dc_model_repo.base import Field, ChartData, Output
from dc_model_repo.base import StepType, FrameworkType, ModelFileFormatType
from dc_model_repo.base.mr_log import logger
from dc_model_repo.step.base import BaseEstimator, ModelWrapperDCStep
from dc_model_repo.step.userdefined_step import UserDefinedEstimator


@six.add_metaclass(abc.ABCMeta)
class Tensorflow2DCStep(ModelWrapperDCStep):
    def __init__(self, origin_model, kind, input_cols, algorithm_name, extension, saver_type=ModelFileFormatType.H5,
                 **kwargs):
        super(Tensorflow2DCStep, self).__init__(self, kind=kind,
                                                framework=FrameworkType.TensorFlow2,
                                                model_format=saver_type,
                                                input_cols=input_cols, algorithm_name=algorithm_name,
                                                extension=extension, **kwargs)
        self.model_path = 'data/model.h5'
        self.origin_model = origin_model
        self.feature_importance = None
        # self.requirements = ["tensorflow==2.1.0"]
        self.limit = 1

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["origin_model", "model", "sample_data"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        model_path = '%s/model.h5' % self.serialize_data_path(destination)
        saved_model_path = os.path.join(self.serialize_data_path(destination), 'model')
        model_path = model_path.replace("\\", "/")  # windows系统转换

        if self.model_format == ModelFileFormatType.H5:
            # 将整个模型保存为HDF5文件
            self.model.save(model_path)
            explanation = [ChartData('netron', 'netron', None, {"path": "data/model.h5"})]
        else:
            if self.saved_model_path is not None:
                fs.copy(self.saved_model_path, saved_model_path)
            explanation = [ChartData('netron', 'netron', None, {"path": "data/model/saved_model.pb"})]
            self.model_path = 'data/model'

        self.explanation = explanation
        self.persist_requirements(fs, destination)

    def persist_requirements(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        from dc_model_repo.util import str_util

        # 生成依赖文件
        from dc_model_repo.util import validate_util
        if validate_util.is_non_empty_list(self.requirements):
            requ_txt = "\n".join(self.requirements)
            fs.make_dirs(self.serialize_requirements_path(destination))
            requ_path = self.serialize_requirements_path(destination) + "/requirements.txt"
            logger.info("依赖文件requirements.txt写入到%s" % requ_path)
            fs.write_bytes(requ_path, str_util.to_bytes(requ_txt))

    def serialize_requirements_path(self, destination):
        return os.path.join(self.serialize_data_path(destination), "requirements")

    def prepare(self, step_path, **kwargs):
        if self.model_format == ModelFileFormatType.H5:
            model_path = "%s/data/model.h5" % step_path
        else:
            model_path = "%s/data/model" % step_path
        logger.info("Prepare to load tensorflow model at: %s" % model_path)
        t1 = time.time()

        ### load model
        # 重新创建完全相同的模型，包括其权重和优化程序
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)

        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("Load tensorflow model ,\ntook %s(s)." % (took))

    

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
        elif isinstance(data, list):
            data = data[0]
            if isinstance(data, np.ndarray):
                return data.dtype
            elif isinstance(data, pd.Series):
                return data.values.dtype

        return ""

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        return self.get_data_sampler(y)

    def get_targets(self, x, y=None, options=None, **kwargs):
        input_features = []
        targets = []

        if not isinstance(x, list):
            x = [x]

        import pandas as pd
        import numpy as np

        for i, name in enumerate(self.input_cols):
            input = x[i]
            if isinstance(input, np.ndarray):
                input_features.append(Field(name, input.dtype, input.shape))
            elif isinstance(input, pd.DataFrame):
                input_features.append(Field(name, None, None))
            else:
                input_features.append(Field(name, None, None))

        for i, name in enumerate(self.output_cols):
            input = y[i]
            if isinstance(input, np.ndarray):
                targets.append(Field(name, input.dtype, input.shape))
            elif isinstance(input, pd.DataFrame):
                columns = input.columns.values.tolist
                if name in columns:
                    data_y = input[name]
                    data = data_y.values
                    targets.append(Field(name, data.dtype, data.shape))
                else:
                    targets.append(Field(name, None, None))
            elif isinstance(input, pd.Series):
                data = input.values
                targets.append(Field(name, data.dtype, data.shape))
            else:
                targets.append(Field(name, None, None))

        self.input_features = input_features
        self.target = targets

        return self.target

    def get_outputs(self, x, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        from dc_model_repo.util import validate_util, operator_output_util

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
                output_fields_label_prob = operator_output_util.get_label_prob_fields(output_field_name, classes_, 'float64')
                # label_prob_fields = [Feature('%s_proba_%s' % (target_name, c), 'float64') for c in classes_]
                #
                if output_field_raw_prob is not None:
                    output_field_raw_prob.shape = [1, len(classes_)]
                else:
                    logger.warning("模型:%s有分类的label:%s， 但是概率为空" % (str(classes_), str(self.model)))
            else:
                logger.warning("模型: %s有分类的label， classes为空。" % str(self.model))
        # max_prob_field=None, raw_prob_field=None, label_prob_fields
        output = Output(output_field_name, output_field_type, y[0].shape, output_field_max_prob, output_field_raw_prob,
                        output_fields_label_prob)
        return [output]

    # def get_model_type(self):
    #     from dc_model_repo.util import cls_util
    #     module_name = cls_util.get_module_name(self.model)
    #     package_array = module_name.split(".")
    #     len_package_array = len(package_array)
    #     part_one = package_array[0]
    #     self.model_type = TensorflowModelType.Keras
    #     if part_one == 'tensorflow_estimator':
    #         self.model_type = TensorflowModelType.Estimator

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
            result_dict.setdefault(outputs[i], predictions[i])

        if hasattr(self.model, 'predict_proba'):
            if calc_max_proba or calc_all_proba:
                proba = self.model.predict_proba(X, batch_size=batch_size, verbose=verbose)
                proba_max = np.amax(proba, axis=1)
                max_prob_field = self.outputs[i].max_prob_field
                if max_prob_field is not None:
                    result_dict.setdefault(self.outputs[i].max_prob_field.name, proba_max)
                else:
                    result_dict.setdefault("%s_%s" % (outputs[i], "max_prob_field"), proba_max)

        return result_dict


class Tensorflow2DCEstimator(BaseEstimator):

    def __init__(self, algorithm_name, framework, model_format=ModelFileFormatType.H5, model=None, 
                model_path=None, input_cols=None, target_cols=None, output_cols=None, extension=None, **kwargs):
        '''
        model: tensorflow2模型对象
        model_path: tensorflow2模型对象序列化后的模型文件，可以为空，如果model_path存在就会直接copy保存，不会再使用模型对象保存；
                    model和model_path不能同时为空。
        '''
        if model is None and model_path is None:
            raise Exception("model 和 model_path 不能同时为空")
        super(Tensorflow2DCEstimator, self).__init__(model=model,
                                                    model_path = model_path,
                                                    algorithm_name=algorithm_name,
                                                    framework=framework,
                                                    model_format=model_format,
                                                    input_cols=input_cols,
                                                    target_cols=target_cols,
                                                    output_cols=output_cols,
                                                    extension=extension, **kwargs)
        self.model = model
        self.model_path = None
        self.input_type = DictDataSampler.DATA_TYPE

    def persist_model(self, fs, destination):
        explanation = None
        if self.model_format == ModelFileFormatType.H5:
            model_path = os.path.join(self.serialize_data_path(destination), 'model.h5')
            if self.model_path is not None:
                fs.copy(self.model_path, model_path)
            else:
                self.model.save(model_path)
            explanation = [ChartData('netron', 'netron', None, {"path": "data/model.h5"})]
            self.model_path = 'data/model.h5'
        elif self.model_format == ModelFileFormatType.SAVED_MODEL:
            model_path = os.path.join(self.serialize_data_path(destination), 'model')
            if self.model_path is not None:
                fs.copy(self.model_path, model_path)
            else:
                self.model.save(model_path)
            explanation = [ChartData('netron', 'netron', None, {"path": "data/model/saved_model.pb"})]
            self.model_path = 'data/model'
        self.explanation = explanation
    
    def prepare(self, step_path, **kwargs):
        if self.model_format == ModelFileFormatType.H5:
            model_path = "%s/data/model.h5" % step_path
        else:
            model_path = "%s/data/model" % step_path
        logger.info("Prepare to load tensorflow model at: %s" % model_path)
        t1 = time.time()

        ### load model
        # 重新创建完全相同的模型，包括其权重和优化程序
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)

        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("Load tensorflow model ,\ntook %s(s)." % (took))
        
    # 判断input是否在model.input_names中，例如：
    # input: dense_input_1, input_names: ['dense_input']
    # 用于兼容6.0.0之前的模型解剖bug
    def is_in_input_names(self, name):
        if name in self.model.input_names: 
            return name, True
        for i in self.model.input_names:
            if i in name:
                return i, True
        return None, False

    def predict(self, X, batch_size=None, verbose=0, steps=None, **kwargs):
        # 传入的数据为dict
        if isinstance(X, dict):
            for k, v in X.items():
                if ':' in k and k.rindex(':') > 0:
                    i_n = k[:k.rindex(':')]
                    i_n, flag = self.is_in_input_names(i_n)
                    if flag:
                        X[i_n] = X.pop(k)
                if isinstance(v, list):
                    v = np.array(v)
        elif isinstance(X, list):
            X = [v for v in X]
        else:
            raise Exception("只支持dict和list类型的输入")
        predictions = self.model.predict(X, batch_size=batch_size, verbose=verbose, steps=steps)
        
        outputs = [o.name for o in self.outputs]
        result_dict = {}
        for i in range(len(outputs)):
            result_dict.setdefault(outputs[i], predictions[i])
        return result_dict

    def transform(self, X):
        return self.predict(X)

    def fit(self, X_train, y_train=None):
        super(Tensorflow2DCEstimator, self).fit(X_train, y_train)

    def get_params(self):
        return super().get_params()
