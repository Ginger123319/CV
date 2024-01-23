# -*- encoding: utf-8 -*-

import abc
import six

from dc_model_repo.base import Field, ChartData, Output
from dc_model_repo.base import StepType, FrameworkType, ModelFileFormatType, TorchDType
from dc_model_repo.base.mr_log import logger
from dc_model_repo.step.base import ModelWrapperDCStep, BaseEstimator


@six.add_metaclass(abc.ABCMeta)
class PytorchDCStep(ModelWrapperDCStep):
    def __init__(self, operator, kind, input_cols, algorithm_name, extension, **kwargs):
        super(PytorchDCStep, self).__init__(kind=kind,
                                            framework=FrameworkType.Pytorch,
                                            model_format=ModelFileFormatType.PTH,
                                            input_cols=input_cols, algorithm_name=algorithm_name,
                                            extension=extension, **kwargs)
        self.model_path = 'data/model.pth'
        self.operator = operator
        self.feature_importance = None
        # self.requirements = ["torch==1.4.0", "torchvision==0.5.0"]
        self.limit = 1
        self.source_code_path = None

        import torch
        self.cuda_available = torch.cuda.is_available()

    def persist_step_self(self, fs, step_path):
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, ["operator", "model"])
        fs.write_bytes(step_path, obj_bytes)

    def persist_model(self, fs, destination):
        model_path = '%s/model.pth' % self.serialize_data_path(destination)
        model_path = model_path.replace("\\", "/")  # windows系统转换

        # 模型解刨，直接copy模型文件
        if hasattr(self, "is_model_dissector") and self.is_model_dissector:
            model_file = self.find_model_file(self.source_code_path)
            fs.copy(model_file, model_path)
        else:
            import torch
            torch.save(self.model, model_path)

        # 1. 将自定义模型目录复制到data下(需要确保data下没有sourceCode目录)。
        if self.source_code_path is not None:
            source_code_dir_path = self.serialize_source_code_path(destination)
            fs.make_dirs(source_code_dir_path)
            if fs.is_dir(self.source_code_path):
                import os
                dir_list = os.listdir(self.source_code_path)
                for element in dir_list:
                    child_path = os.path.join(self.source_code_path, element)
                    if fs.is_file(child_path):
                        fs.copy(self.join_path(self.source_code_path, element),
                                self.join_path(source_code_dir_path, element))
            else:
                source_code_path = self.join_path(source_code_dir_path, self.source_code_path.split('\\')[-1])
                logger.info("源码目录复制到: %s" % source_code_path)
                fs.copy(self.source_code_path, source_code_path)

        explanation = [ChartData('pytorch', 'netron', None, {"path": "data/model.pth"})]
        self.explanation = explanation
        self.persist_requirements(fs, destination)

    def serialize_source_code_path(self, destination):
        return self.join_path(self.serialize_data_path(destination), "sourceCode")

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
        import os
        return os.path.join(self.serialize_data_path(destination), "requirements")

    def prepare(self, step_path, **kwargs):
        model_path = "%s/data/model.pth" % step_path
        sourceCode = "%s/data/sourceCode" % step_path
        logger.info("Prepare to load pytorch model at: %s" % model_path)
        import os
        import time
        import torch

        t1 = time.time()

        if os.path.exists(sourceCode):
            if os.path.isdir(sourceCode):  # 加载插件
                dir_list = self.get_model_file(sourceCode)
                if len(dir_list) > 0:
                    from pytorch_model_import import ModuleImporter

                    model_file = self.find_model_file(sourceCode)
                    modelImporter = ModuleImporter()
                    self.model = modelImporter.load(model_file)
                else:
                    self.model = torch.load(model_path)
            else:  # 加载pkl
                self.model = torch.load(model_path)
        else:
            self.model = torch.load(model_path)

        self.to_devices(self.model)

        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("Load Pytorch model ,\ntook %s(s)." % (took))

    def get_params(self):
        """
        解析pytorch模型的参数
        :return:
        """
        from dc_model_repo.base import Param
        params = []
        for k in self.input_cols:
            params.append(Param(k, None, None))

        # 增加输入列
        params.append(Param('input_cols', None, self.input_cols))
        return params

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
        if data is None:
            return ""

        if isinstance(data, np.ndarray):
            return data.dtype.name
        elif isinstance(data, pd.DataFrame):
            return list(data.dtypes.to_dict().values())[0].name
        elif isinstance(data, list) and len(data) > 0:
            data = data[0]
            if isinstance(data, np.ndarray):
                return data.dtype

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
            input = x[i]
            if isinstance(input, np.ndarray):
                targets.append(Field(name, input.dtype, input.shape))
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
        y_shape = None
        if y is not None and len(y) > 0:
            y_shape = y[0].shape

        output = Output(output_field_name, output_field_type, y_shape, output_field_max_prob, output_field_raw_prob,
                        output_fields_label_prob)
        return [output]

    def predict_result(self, X, predictions, calc_max_proba=True, calc_all_proba=False):
        if not isinstance(X, list):
            # X = [X]
            predictions = [predictions]

        outputs = [o.name for o in self.outputs]
        result_dict = {}
        for i in range(0, len(outputs)):
            result_dict.setdefault(outputs[i], predictions[i])

        return result_dict

    def get_model_file(self, model_path):
        import os
        dir_list = os.listdir(model_path)
        dir_list = sorted(dir_list, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
        dir_list = [element for element in dir_list if
                    element.endswith(".pth") or element.endswith(".pt") or element.endswith(".pkl")]
        return dir_list

    def find_model_file(self, model_path):
        import os
        dir_list = self.get_model_file(model_path)
        if len(dir_list) == 0:
            raise Exception("无法识别文件，未找到.pth文件，路径：{0}。".format(model_path))
        dir_list = dir_list[::-1]
        return os.path.join(model_path, dir_list[0])

    def change_tensor_type(self, data, torchType):
        import torch
        torchType = str(torchType)
        if torchType is None:
            data = torch.from_numpy(data)
        elif torchType.startswith(TorchDType.Float):  # float、float16、float32
            data = torch.from_numpy(data).float()
        elif torchType.startswith(TorchDType.Float):
            data = torch.from_numpy(data).long()
        elif torchType.startswith(TorchDType.Bool):
            data = torch.from_numpy(data).bool()
        elif torchType.startswith(TorchDType.Byte):
            data = torch.from_numpy(data).byte()
        elif torchType.startswith(TorchDType.Char):
            data = torch.from_numpy(data).char()
        elif torchType.startswith(TorchDType.Int):
            data = torch.from_numpy(data).int()
        elif torchType.startswith(TorchDType.Short):
            data = torch.from_numpy(data).short()

        return data

    def to_devices(self, x):
        if not self.cuda_available:
            return x

        if len(self.cuda_device_ids) == 0:
            return x

        device_cuda = "cuda:%s" % self.cuda_device_ids[0]
        logger.info("data to torch.device[%s]", device_cuda)
        return x.cuda(device=self.cuda_device_ids[0])


class PytorchDCEstimator(PytorchDCStep, BaseEstimator):

    def __init__(self, model, algorithm_name=None, source_code=None, **kwargs):
        # 2. 调用父类构造方法
        super(PytorchDCEstimator, self).__init__(operator=model, model=model, kind=StepType.Estimator,
                                                 input_cols=None, algorithm_name=algorithm_name,
                                                 extension=None, **kwargs)

        self.model = model
        self.source_code_path = source_code
        self.cuda_device_ids = [0]  # gpu device

    def predict(self, X, batch_size=None,
                verbose=0,
                steps=None, calc_max_proba=True, calc_all_proba=False, **kwargs):
        inputs = [i.name for i in self.input_features]
        inputTypes = [i.type for i in self.input_features]
        import numpy as np

        model_params = []

        def change_data_type(self, X):
            import numpy as np
            for index, key in enumerate(inputs):
                if isinstance(X, dict):
                    data = X[key]
                elif isinstance(X, list):
                    data = X[index]
                else:
                    data = X
                dataType = inputTypes[index]
                if isinstance(data, list):
                    data = self.change_tensor_type(np.array(data), dataType)
                elif isinstance(data, np.ndarray):
                    data = self.change_tensor_type(data, dataType)

                data = self.to_devices(data)
                model_params.append(data)

        # 模型部署时传入的数据为dict，ndarray为list
        if isinstance(X, dict):
            change_data_type(self, X)
        elif isinstance(X, list):
            change_data_type(self, X)
        elif isinstance(X, np.ndarray):
            data = self.change_tensor_type(X, inputTypes[0])
            data = self.to_devices(data)
            model_params.append(data)
        else:
            X = self.to_devices(X)
            model_params.append(X)

        pre = self.model(*model_params)
        if self.cuda_available and len(self.cuda_device_ids) > 0:
            predictions = pre.cuda().data.cpu().numpy()
        else:
            predictions = pre.data.numpy()

        return self.predict_result(X, predictions, calc_max_proba=calc_max_proba, calc_all_proba=calc_all_proba)

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

        if y_train is None:
            y_train = []

        super(PytorchDCEstimator, self).fit(X_train, y_train)
