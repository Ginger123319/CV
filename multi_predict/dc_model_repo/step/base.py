# -*- encoding: utf-8 -*-
import os
from os import path as P

import abc
import pandas as pd
import six
import uuid

from dc_model_repo.base import BaseOperator, StepType, DictSerializable, Field, TrainInfo, FrameworkType, BaseOperatorMetaData, Param, ChartData, Output, LearningType, DatasetType
from dc_model_repo.util import cls_util, str_util, validate_util, json_util, dataset_util, pkl_util


class StepMetaData(DictSerializable, BaseOperatorMetaData):
    """使用这个类来封装Step的相关信息，pipeline模型文件里各个step里的meta.json是从这个类的实例序列化来的

    Args:
        id: 执行 :meth:`__init__` 时生成的 uuid.
        module_id: 在工作流中当前 step 所在的模块 id
        name: 算法名称
        version: sdk 版本号
        step_type: step 类型。 Refer to: :class:`dc_model_repo.base.StepType`
        timestamp: 毫秒时间戳
        class_name: Step 的类名
        algorithm_name: 算法名称
        framework: 训练框架的类型。Refer to :class:`dc_model_repo.base.FrameworkType`
        model_format: 模型的文件格式。Refer to :class:`dc_model_repo.base.ModelFormatType`
        model_path: 模型路径。一般为None
        language: six包中python语言版本
        input_features (list): 进入 pipeline 的输入列信息。 list of :class:`dc_model_repo.base.Field`
        params:
        input_type: 输入 step 的数据的类型
        output_type: 输出 step 的数据的类型
        target: 目标列。 list of :class:`dc_model_repo.base.Field`
        outputs: 输出列。 list of :class:`dc_model_repo.base.Field`
        train_info: 训练时间、数据行列等信息。
        attachments: 模型附件
        extension (dict): 扩展信息，包括构造 pipeline 所在算子所处的环境信息。
        extra (dict): 目前一般为 `{}`
    """

    def __init__(self, id, module_id, name, version, step_type, timestamp, class_name, algorithm_name, framework, model_format,
                 model_path, language, input_features, params, input_type, output_type,
                 target=None, outputs=None, train_info=None, attachments=None, extension=None, extra=None):

        super(StepMetaData, self).__init__(module_id)
        self.id = id
        self.name = name
        self.version = version
        self.step_type = step_type
        self.timestamp = timestamp
        self.class_name = class_name
        self.algorithm_name = algorithm_name
        self.framework = framework
        self.model_format = model_format
        self.model_path = model_path
        self.language = language
        self.input_features = input_features
        self.input_type = input_type
        self.output_type = output_type
        # self.sample_data_path = sample_data_path
        # self.target_sample_data_path = target_sample_data_path
        self.target = target
        self.outputs = outputs
        # {"trainTime":2.462,"trainSetCols":1,"trainSetRows":7000,"testSetCols":1,"testSetRows":1400}
        self.train_info = train_info
        self.params = params
        self.attachments = attachments
        self.extension = extension
        self.extra = extra

    @classmethod
    def field_mapping(cls):
        return {'id': 'id',
                'module_id': 'moduleId',
                'name': 'name',
                'version': 'version',
                'step_type': 'stepType',
                'timestamp': 'timestamp',
                'class_name': 'className',
                'algorithm_name': 'algorithmName',
                'framework': 'framework',
                'model_format': 'modelFormat',
                'model_path': 'modelPath',
                'language': 'language',
                'input_type': 'inputType',
                'output_type': 'outputType',
                # 'sample_data_path': 'sampleDataPath',
                # 'target_sample_data_path': 'targetSampleDataPath',
                # 'train_info': 'trainInfo',
                'attachments': 'attachments',
                'extension': 'extension',
                'extra': 'extra'
                }

    def to_dict(self):
        result_dict = self.member2dict(self.field_mapping())
        result_dict['inputFeatures'] = None if self.input_features is None else [ep.to_dict() for ep in self.input_features]
        result_dict['target'] = None if self.target is None else [ep.to_dict() for ep in self.target]
        result_dict['outputs'] = None if self.outputs is None else [ep.to_dict() for ep in self.outputs]
        result_dict['params'] = None if self.params is None else [ep.to_dict() for ep in self.params]
        result_dict['trainInfo'] = DictSerializable.to_dict_if_not_none(self.train_info)
        return result_dict

    @staticmethod
    def len_construction_args():
        return 21

    @classmethod
    def load_from_dict(cls, dict_data):
        # 1. 填充简单Key
        bean = super(StepMetaData, cls).load_from_dict(dict_data)

        # 2. 设置复杂key
        bean.input_features = Field.load_from_dict_list(dict_data['inputFeatures'])
        bean.target = Field.load_from_dict_list(dict_data['target'])
        bean.outputs = Output.load_from_dict_list(dict_data['outputs'])
        bean.params = Param.load_from_dict_list(dict_data['params'])
        bean.train_info = TrainInfo.load_from_dict(dict_data['trainInfo'])

        return bean


@six.add_metaclass(abc.ABCMeta)
class DCStep(BaseOperator):
    """APS 对处理步骤的抽象，包括对数据进行转换的Transformer和进行评估的Estimator两种类型。

    Args:
        operator(object): 被包装的算法模型。
        framework (str): 训练框架的类型。Refer to :class:`dc_model_repo.base.FrameworkType`
        model_format (str): 模型的文件格式，Refer to :class:`dc_model_repo.base.ModelFileFormatType`
        input_cols (list): 输入列。
        algorithm_name (str): 算法名称，如果为空则会从第三方模型中推测。
        explanation (list): 模型可解释数据，需要是 :class:`dc_model_repo.base.ChartData`类型的数组。
        extension (dict): 扩展信息，之后会向其中添加工作流中当前step所在算子所处的环境信息。
        source_code_path: 第三方模型所依赖的源码文件（夹）。
        requirements: 依赖外部库，例如：["requests==0.1"]，会将其记录到step持久化文件中的requirements.txt
        sample_limit: 对输入数据的采样行数
        output_type: 指定的输出类型。如果为None，后续程序执行fit时会设置成跟input_type一致的。
        **kwargs: 向后传播的参数。
    """

    # 常量定义
    FILE_REQUIREMENTS = "requirements"
    FILE_SOURCE_CODE = "sourceCode"

    def __init__(self, operator,
                 framework,
                 model_format,
                 input_cols,
                 algorithm_name=None,
                 explanation=None,
                 source_code_path=None,
                 requirements=None,
                 extension=None,
                 sample_limit=1,
                 output_type=None,
                 **kwargs):

        super(DCStep, self).__init__(str(uuid.uuid4()), extension)
        self.operator = operator  # 训练前的模型
        self.model = None  # 训练后的模型, fit之后赋值

        if algorithm_name is None:
            self.algorithm_name = cls_util.get_class_name(operator)

        # transformer, estimator
        self.framework = framework
        self.model = None  # step 内部的模型
        self.input_cols = input_cols

        self.algorithm_name = algorithm_name
        self.explanation = explanation  # 类型为List[ChartData]

        self.sample_limit = sample_limit

        self.input_features = None

        self.input_type = None
        if output_type is not None and output_type not in DatasetType.all_values():
            raise Exception("当前支持的输出类型为：{} 暂不支持：[{}]".format(repr(DatasetType.all_values()), repr(output_type)))
        self.output_type = output_type
        self.model_path = None

        self.source_code_path = source_code_path  # 额外依赖的源码路径

        self.model_format = model_format
        # self.output_cols = {}
        # 样本数据落地到单独的文件中，并且保存到对象内部，否则无法自动读取pipeline的input_features
        self.sample_data = None
        self.attachments = []
        self._fitted = False
        self.params = None  # 获取模型参数通过此属性而不是 get_params，因为get_params需要现解析
        self.extra = {}

        self.requirements = requirements

    def serialize_step_path(self, destination):
        return '%s/step.pkl' % self.serialize_data_path(destination)

    def serialize_explanation_path(self, destination):
        return '%s/explanation' % destination

    def serialize_explanation_meta_path(self, destination):
        return '%s/explanation.json' % self.serialize_explanation_path(destination)

    def serialize_explanation_attachments_path(self, destination):
        return '%s/attachments' % self.serialize_explanation_path(destination)

    def serialize_requirements_path(self, destination):
        return BaseOperator.join_path(self.serialize_data_path(destination), self.FILE_REQUIREMENTS)

    @staticmethod
    def serialize_source_code_path(destination):
        return BaseOperator.join_path(BaseOperator.serialize_data_path(destination), DCStep.FILE_SOURCE_CODE)

    def get_feature_sample_data(self, X):
        """Alias of :meth:`get_data_sampler`"""
        return self.get_data_sampler(X)

    def fit_prepare(self, X, y=None, options=None, **kwargs):
        """实际训练前的hook。参数同 :meth:`fit`

        这里修改过的参数和X，会用于实际的训练

        Returns: X
        """
        return X

    def fit_post(self, X, y=None, options=None, **kwargs):
        """实际训练后的hook。参数同 :meth:`fit`"""
        pass

    def estimator_logic(self, X, y=None, options=None, **kwargs):
        """将Estimator的后处理逻辑挪到BaseEstimator里"""
        pass

    # @abc.abstractmethod
    def fit(self, X, y=None, options=None, **kwargs):
        """训练模型。

        Args:
            X (Pandas DataFrame, PySpark DataFrame or Dict): 训练用特征数据。
            y : 训练用标签数据。
            options(dict): 送给 :attr:`operator` 的 ``fit`` 方法的参数。
            **kwargs: 扩展字段。

        Returns:
            返回self。
        """
        from dc_model_repo.base.mr_log import logger

        # 1. 调用fit前准备(fit_prepare是最先执行)
        X = self.fit_prepare(X, y, options, **kwargs)

        # 2. 抓取feature样本数据
        self.sample_data = self.get_feature_sample_data(X)

        # 3. 检测输入的数据的格式
        self.input_type = self.sample_data.get_data_type()

        # 4. 设置输入的schema
        # 将获取input_features的逻辑转移到data sampler中
        self.input_features = self.sample_data.get_input_features(self.input_cols)

        # 5. 模型fit方法(收集样本数据，schema等信息后开始训练模型)
        if options is None:
            options = {}
        import time
        begin_time = time.time()
        self.model = self.fit_model(X, y, options, **kwargs)  # 给model赋值
        end_time = time.time()
        elapsed = round((end_time - begin_time) * 1000, 3)

        logger.info("训练消耗%s毫秒, 模型信息:\n%s" % (elapsed, str(self.model) if self.model is not None else ""))
        self.elapsed = elapsed

        # 6. 解析参数
        self.params = self.get_params()
        if self.params is None or len(self.params) == 0:
            logger.warning("当前DCStep的params信息为空，如果为自定义DCStep，可以覆盖实现get_params方法提供参数信息。")

        # 7. 处理Estimator相关的信息
        self.estimator_logic(X, y, options, **kwargs)

        # 8. 调用自定义fit后处理
        self.fit_post(X, y, options, **kwargs)

        # 9. 设置已经调用过 fit
        self._fitted = True

        # 如果为None，设置成与输入类型一致。如果不为None，说明在初始化DCStep时候，已经主动设置过了output_type了，就不再干预了。
        if self.output_type is None:
            self.output_type = self.input_type

        return self

    def persist_prepare(self, fs, destination):
        """持久化前的准备

        创建序列化目录以及下面的data目录。复制 ``self.source_code_path`` 指定的源码到序列化目录 ``data/sourceCode/{sourceFiles}`` 。

        Args:
            fs: 当前运行时存放模型的文件管理器代理，为 :class:`dc_model_repo.base.file_system.DCFileSystem` 的实现类。
                通常:

                - 分布式环境为 :class:`dc_model_repo.base.file_system.HDFSFileSystem`，
                - 单机环境为 :class:`dc_model_repo.base.file_system.LocalFileSystem`

            destination: 当前运行时保存模型文件的路径

        Examples:
            Case 1::

                如果 source_code_path = 'my_step.py'，复制结果为:
                data
                  |-sourceCode
                       |-my_step.py

            Case 2::

                如果 source_code_path = 'my_step'，my_step的目录结构为:
                |-my_step
                   | my_step.py
                复制结果为：
                data
                  |-sourceCode
                       |-my_step
                            |-my_step.py
        """

        # 1.检查此目录下是否已经存在当前ID
        if not fs.exists(destination):
            # 2.创建data目录
            p_data = self.serialize_data_path(destination)
            fs.make_dirs(p_data)

        if self.source_code_path is not None:
            self._copy_source_code_into_model(fs, self.source_code_path, self.serialize_source_code_path(destination))

    @staticmethod
    def get_persist_path():
        from dc_model_repo.base import collector_manager
        extension_collector = collector_manager.get_extension_collector()
        if extension_collector is not None:
            return extension_collector.get_steps_path()
        else:
            return "./steps"  # 本地测试生成到steps目录中

    def persist_step_self(self, fs, step_path):
        """持久化当前 `DCStep` 对象

        Args:
            fs: 当前运行时存放模型的文件管理器代理，为 :class:`dc_model_repo.base.file_system.DCFileSystem` 的实现类。
                通常:

                - 分布式环境为 :class:`dc_model_repo.base.file_system.HDFSFileSystem`，
                - 单机环境为 :class:`dc_model_repo.base.file_system.LocalFileSystem`

            step_path: 保存pkl文件的路径
        """
        # 1. 确保目录存在
        dir_path = P.dirname(step_path)
        if not fs.exists(dir_path):
            fs.make_dirs(dir_path)

        # 2. 序列化step
        variables = self.get_persist_step_ignore_variables()
        from dc_model_repo.base.mr_log import logger
        if validate_util.is_non_empty_list(variables):
            logger.info("序列化Step忽略其中的属性:%s" % ",".join(variables))
        obj_bytes = pkl_util.serialize_with_ignore_variables(self, variables)
        fs.write_bytes(step_path, obj_bytes)

    @abc.abstractmethod
    def persist_model(self, fs, destination):
        """序列化模型训练的结果， 适用于第三方模型落地。

        Args:
            fs: 当前运行时存放模型的文件管理器代理
            destination: 保存模型的路径

        """
        pass

    def load_model(self, step_path):
        """ 加载模型。
        Args:
            step_path: 模型文件路径

        Returns:
            返回模型对象。
        """
        pass

    def persist_post(self, fs, destination):
        """序列化后置处理。"""
        pass

    def persist_explanation(self, fs, destination):
        """序列化模型训练的结果， 适用于第三方模型落地。"""
        pass

    def get_persist_step_ignore_variables(self):
        """定义序列化step时候需要忽略的属性。

        Returns:
            list: 返回字符串数组。
        """
        return ["model", "operator"]

    def get_default_persist_destination(self):
        return "%s/%s" % (DCStep.get_persist_path(), self.id)

    def to_meta(self):
        """ 把当前DCStep对象转换元数据。

        Returns:
            dc_model_repo.step.base.StepMetaData: DCStep的元数据对象。
        """
        from dc_model_repo import __version__
        from dc_model_repo.util import time_util

        # 1. 检查目标变量
        if isinstance(self, BaseEstimator) and self.learning_type != LearningType.Clustering:
            if self.target is None or len(self.target) == 0:
                raise Exception("Estimator的目标信息不能为空。")

        # 2. 生成输入特征变量
        if isinstance(self, BaseEstimator):
            kind = StepType.Estimator
        else:
            kind = StepType.Transformer

        meta_data = StepMetaData(id=self.id, module_id=self.module_id, name=self.algorithm_name,
                                 version=__version__, step_type=kind, timestamp=time_util.current_time_millis(),
                                 class_name=cls_util.get_full_class_name(self), algorithm_name=self.algorithm_name,
                                 framework=self.framework, model_format=self.model_format, model_path=self.model_path,
                                 language=cls_util.language(), input_features=self.input_features,
                                 input_type=self.input_type,
                                 output_type=self.output_type,
                                 params=self.params, attachments=self.attachments,
                                 extension=self.extension, extra=self.extra)

        # 3. 追加estimator的属性 TODO k 针对estimator的处理挪到BaseEstimator中
        if isinstance(self, BaseEstimator):
            meta_data.target = self.target
            meta_data.outputs = self.outputs
            meta_data.train_info = self.train_info

        return meta_data

    # @abc.abstractmethod
    def transform(self, X, **kwargs):
        """转换数据。

        Args:
            X (object): 训练的特征数据, 可以是pandas或者PySpark的DataFrame或者np.ndarray。

        Returns:
          返回转换后的数据，格式与输入的X的格式保持相同。
        """
        # 1. 校验输入的类型是否匹配并去除无关数据
        remove_unnecessary_cols = kwargs["remove_unnecessary_cols"] if "remove_unnecessary_cols" in kwargs else False
        X = dataset_util.validate_and_cast_input_data(X, self.input_type, self.input_features, remove_unnecessary_cols=remove_unnecessary_cols)

        # 2. 转换数据
        return self.transform_data(X, **kwargs)

    def persist(self, destination=None, fs_type=None, persist_sample_data=False, persist_explanation=True, **kwargs):
        """持久化DCStep到指定路径

        Args:
            destination (str): 当前Step要持久化到的路径
            fs_type (str): 文件系统类型，一般不填。会通过 :meth:`dc_model_repo.base.BaseOperator.get_fs_type` 推断。
            persist_sample_data (bool): 是否持久化样本数据
            persist_explanation (bool): 是否持久化模型解释数据。
            **kwargs: 备用
        """
        if destination is None:
            destination = self.get_default_persist_destination()
        # 1. 推断文件系统的类型
        if fs_type is None:
            fs_type = self.get_fs_type()

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(fs_type)

        from dc_model_repo.base.mr_log import logger
        logger.info('文件系统类型是: %s, 准备序列化当前Step到[%s]. 当前Step信息:\n[%s]' % (fs_type, destination, str(self)))

        # 2. 检查是否fit过
        if not self._fitted:
            raise Exception("当前Step没有fit过， 无法生成样本数据和Step的输入Schema.")

        # 3. 准备目录
        self.persist_prepare(fs, destination)

        # 4. 序列化原始模型（这个是不同的部分抽象出去自定义）
        self.persist_model(fs, destination)

        # 5. 序列化Step对象（DCStep内部的某些对象无法持久化，需要单独处理）
        self.persist_step_self(fs, self.serialize_step_path(destination))

        # 6. 生成依赖文件
        if validate_util.is_non_empty_list(self.requirements):
            requ_txt = "\n".join(self.requirements)
            fs.make_dirs(self.serialize_requirements_path(destination))
            requ_path = self.serialize_requirements_path(destination) + "/requirements.txt"
            logger.info("依赖文件requirements.txt写入到%s" % requ_path)
            fs.write_bytes(requ_path, str_util.to_bytes(requ_txt))

        # 7.序列化meta数据
        meta_data_json_path = self.serialize_meta_path(destination)
        meta_data_json = self.to_meta().to_json_string()
        fs.write_bytes(meta_data_json_path, str_util.to_bytes(meta_data_json))

        # 8. 序列化样本数据
        self.persist_sample_data(fs, destination, persist_sample_data)

        # 9. 序列化模型解释数据
        if persist_explanation:
            # 8.1. 如果在这之前已经指定了可视化对象，那么直接序列化(explanation对象序列化可能会被忽略)
            if self.explanation is not None:
                logger.info("用户设置了解释数据，不再从persist_explanation方法中解析，将这些数据落地。")
                self.persist_explanation_object(fs, destination, self.explanation)
            else:
                # 8.2. 没有设置可视化对象，调用方法解析解释数据
                self.persist_explanation(fs, destination)
        else:
            logger.info("已设置跳过序列化模型解释数据。")

        # 10. 序列化后置处理
        self.persist_post(fs, destination)

    def persist_explanation_object(self, fs, destination, explanation):
        # 1. 创建文件夹
        fs.make_dirs(self.serialize_explanation_path(destination))

        # 2. 序列化数据
        fs.write_bytes(self.serialize_explanation_meta_path(destination), json_util.to_json_bytes([e.to_dict() for e in explanation]))

    # @abc.abstractmethod
    def prepare(self, step_path, **kwargs):
        """预加载模型。

        反序列化时，如果模型依赖外部文件或者数据可以在此方法中读取，执行完毕后，DCStep实例可以用来预测数据。

        Args:
            step_path: Step的目录。
            **kwargs: 备用
        """
        self.model = self.load_model(step_path)

    @abc.abstractmethod
    def get_params(self):
        """解析第三方模型的参数。

        如果是反序列化的Step，必须在prepare方法调用之后才能调用此方法。如果要获取模型参数，建议通过params属性获取。

        Returns:
          list: 返回模型的参数, 数组内元素的类型为:class:`dc_model_repo.base.Param`
        """
        pass

    # @abc.abstractmethod
    def get_data_sampler(self, data):
        """设置数据抽样器。

        Args:
            data: 数据

        Returns:
            dc_model_repo.pipeline.pipeline.DataSampler: 样本抽样器。
        """
        if data is None:
            raise Exception("输入数据不能为空。")
        from dc_model_repo.base.data_sampler import get_appropriate_sampler
        return get_appropriate_sampler(X=data, limit=self.sample_limit)

    def fit_input_model(self, input_model, X, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        if hasattr(input_model, 'fit'):
            if y is None:
                return input_model.fit(X, **options)
            else:
                return input_model.fit(X, y, **options)
        else:
            logger.info("对象model没有fit方法, obj=%s" % str(input_model))
            return input_model

    def fit_model(self, X, y=None, options=None, **kwargs):
        """使用原始模型训练并返回最终可以使用的模型。参数同 :meth:`fit`

        Returns: 对 ``operator`` 调用 ``fit`` 后的结果。
        """
        return self.fit_input_model(self.operator, X, y, options, **kwargs)

    def get_params_from_dict_items(self, dict_items, input_cols):
        """解析SKLearn模型的参数， 使用训练前的原始模型。"""

        def is_param(k, v):
            if not isinstance(k, str):
                return False
            if len(k) < 1:
                return False
            if k[0] == "_":
                return False
            if isinstance(v, dict):
                return False
            return True

        params = []
        for k in dict_items:
            v = dict_items[k]
            if is_param(k, v):
                params.append(Param(k, None, str(v)))

        # 增加输入列
        params.append(Param('input_cols', None, str(input_cols)))
        return params

    def transform_data(self, X, **kwargs):
        """对数据进行转换。"""

        return X

    @staticmethod
    def load(path, fs_type=None):
        """静态方法，从文件系统中反序列化DCStep。

        Args:
            path: step的目录地址。
            fs_type (str): 文件系统类型，一般不填。会通过 :meth:`dc_model_repo.base.BaseOperator.get_fs_type` 推断。

        Returns:
            DCStep实例。
        """

        # 1. 推断文件系统的类型
        if fs_type is None:
            fs_type = BaseOperator.get_fs_type()
        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(fs_type)

        # 2. 如果有单机自定义的，把源码文件加载到classpath中
        # 2.1. 加载meta文件
        # step_meta = StepMetaData.load_from_json_str(str_util.to_str(fs.read_bytes(BaseOperator.serialize_meta_path(path))))
        # 把源码目录加载到classpath
        source_code_path = DCStep.serialize_source_code_path(path)
        import sys
        if fs.exists(source_code_path):
            print("加载PYTHONPATH: %s" % source_code_path)
            sys.path.append(source_code_path)

        # 3. 反序列化step.pkl
        p_step_pkl = "%s/data/step.pkl" % path
        step_bytes = fs.read_bytes(p_step_pkl)
        from dc_model_repo.util import pkl_util
        step = pkl_util.deserialize(step_bytes)

        # 4. 准备激活step
        # step.prepare(path) # 用户需要手动激活.
        return step

    def get_model(self):
        """返回 ``self.model`` """
        return self.model

    def get_operator(self):
        """返回 ``self.operator`` """
        return self.operator

    def __str__(self):
        _cls_name = cls_util.get_full_class_name(self)
        return "Step class=%s, id=%s, algorithm_name=%s, model=%s" % (_cls_name, str(self.id), self.algorithm_name, str(self.model))

    @staticmethod
    def _delete_py_cache_and_bin_files(fs, dir_path):
        from dc_model_repo.base.mr_log import logger
        sub_files = fs.listdir(dir_path)
        for file_name in sub_files:
            file_path = os.path.join(dir_path, file_name)
            if file_name == "__pycache__" and fs.is_dir(file_path):
                #  条件： 名称是__pycache__，并且是个文件夹
                logger.info("删除Python缓存目录: %s" % file_path)
                fs.delete_dir(file_path)
            elif fs.is_file(file_path) and len(file_name) >= 4 and file_name[-4:] == ".pyc":
                # 条件：是个文件，并且后缀是.pyc
                logger.info("删除Python pyc文件: %s" % file_path)
                fs.delete_file(file_path)
            else:
                if fs.is_dir(file_path):
                    # 如果是目录，并且不是__pycache__ 就递归删除
                    DCStep._delete_py_cache_and_bin_files(fs, file_path)

    @staticmethod
    def _copy_source_code_into_model(fs, source_code_path, dest_source_code_dir_path):
        from dc_model_repo.base.mr_log import logger
        fs.make_dirs(dest_source_code_dir_path)  # 确保sourceCode目录存在
        # fixmeDone P.basename("dir/") 得到的值为 /  在_fix_source_code_path方法中做了fix
        dest_source_code_dir_path = P.join(dest_source_code_dir_path, P.basename(source_code_path))
        logger.info("正在复制源码...\n %s --> %s" % (source_code_path, dest_source_code_dir_path))
        fs.copy(source_code_path, dest_source_code_dir_path)  # 复制文件到源码目录

        logger.info("正在清理源码中的缓存文件...")
        # 3. 清理源码中的pyc文件和__pycache__目录
        if fs.is_dir(dest_source_code_dir_path):
            DCStep._delete_py_cache_and_bin_files(fs, dest_source_code_dir_path)
        logger.info("复制完成")

@six.add_metaclass(abc.ABCMeta)
class BaseTransformer(DCStep):
    """所有Transformer的基类"""

    pass


@six.add_metaclass(abc.ABCMeta)
class BaseEstimator(DCStep):
    """所有Estimator的基类。

    初始化参数在DCStep基础上增加了output_cols, target_cols, learning_type

    Args:
        output_cols: 输出列，list类型，如果为None或[]，会设置成默认值 ``["prediction"]``
          这个属性会通过get_outputs转为self.outputs，默认的get_outputs只支持一个元素，如果需要输出多列，需要复写get_outputs方法。
        target_cols: 标签列，list类型，如果为None或[]，会设置成默认值 ``["label"]``
          这个属性会通过get_targets转为self.target,默认的get_targets只支持一个元素，如果需要支持多列，需要复写get_targets方法。
        learning_type: 学习类型，标识该模型是分类、回归还是聚类，如果为None，会设置成默认值 :attr:`dc_model_repo.base.LearningType.Unknown`
          可选值见：:class:`dc_model_repo.base.LearningType`
        operator: 传入的算子，会对这个算子执行 ``.fit(X, y, **options)``
        framework: 训练框架的类型，见 :class:`dc_model_repo.base.FrameworkType`
        model_format: 模型的文件格式，见 :class:`dc_model_repo.base.ModelFormatType`
        input_cols: 当前算子要处理的列，在fit过程会取X的列与其交集当设置到 :attr:`input_features`
        algorithm_name: 算法名称
        explanation (list): 模型可解释数据，需要是 :class:`dc_model_repo.base.ChartData` 类型的数组。
        source_code_path: 自定义算子的源码路径
        requirements: 依赖的外部库，例如：["requests==0.1"]
        extension: 扩展信息字段
        **kwargs: 预留参数位置
    """

    ALLOWED_OUTPUT_TYPE = (DatasetType.PandasDataFrame, DatasetType.PySparkDataFrame, DatasetType.Dict)

    def __init__(self,
                 output_cols=None,
                 target_cols=None,
                 operator=None,
                 framework=None,
                 model_format=None,
                 input_cols=None,
                 algorithm_name=None,
                 explanation=None,
                 source_code_path=None,
                 requirements=None,
                 extension=None,
                 learning_type=None,
                 **kwargs):

        from dc_model_repo.base.mr_log import logger

        if target_cols is None or len(target_cols) == 0:
            target_cols = ["label"]
            logger.warning("您没有设置target_cols属性，将使用默认值['label']，与实际训练数据集可能不一致，请您尽快完善自定义的Estimator准确设置该属性。")

        if output_cols is None or len(output_cols) == 0:
            output_cols = ['prediction']
            logger.warning("您没有设置output_cols属性，将使用默认值['prediction']，与实际预测输出的列可能不一致，请您尽快完善自定义的Estimator准确设置该属性。")

        if learning_type is None:
            learning_type = LearningType.Unknown

        super(BaseEstimator, self).__init__(operator=operator,
                                            framework=framework,
                                            model_format=model_format,
                                            input_cols=input_cols,
                                            algorithm_name=algorithm_name,
                                            explanation=explanation,
                                            source_code_path=source_code_path,
                                            requirements=requirements,
                                            extension=extension,
                                            **kwargs)

        self.target_cols = target_cols  # 目标列, 4.0后必须填写
        self.output_cols = output_cols  # 输出列, 4.0后必须填写
        self.learning_type = learning_type  # 机器学习的算法类型

        self.feature_importance = None  # 特征重要性
        self.train_columns = None  # 训练列，给特征重要性生成列名
        self.outputs = None  # 描述输出列的信息，包含概率列的信息
        self.train_info = None  # 训练数据集信息
        self.target = None  # 标签列信息
        self.train_data_shape = None  # 训练数据集的信息

    def fit(self, X, y=None, options=None, **kwargs):
        super(BaseEstimator, self).fit(X=X, y=y, options=options, **kwargs)

        # 为了满足模型服务需要，校验输出类型
        if self.output_type not in BaseEstimator.ALLOWED_OUTPUT_TYPE:
            raise Exception("Estimator的输出类型output_type必须是{}中的一个，现在却为：{}。可在初始化Step时使用参数output_type明确指定!".format(repr(BaseEstimator.ALLOWED_OUTPUT_TYPE), repr(self.output_type)))
        return self

    def estimator_logic(self, X, y=None, options=None, **kwargs):
        # 7. 处理Estimator相关的信息
        from dc_model_repo.base.mr_log import logger
        #if isinstance(self, BaseEstimator):
        # 7.1. 计算Target和其样本数据
        self.target_sample_data = self.get_target_sample_data(X, y, options, **kwargs)

        # 7.2. 记录训练数据集信息
        train_set_rows = None
        train_set_cols = None
        from pyspark.sql import DataFrame
        if hasattr(X, 'shape'):
            train_set_rows = X.shape[0]
            train_set_cols = X.shape[1]

        elif isinstance(X, DataFrame):
            train_set_rows = X.count()
            # fixme: spark 的序列化不准，封装成向量了。
            train_set_cols = len(X.columns)
        self.train_info = TrainInfo(train_set_rows, train_set_cols, self.elapsed)

        # 7.3. 获取目标列
        target = self.get_targets(X, y, options, **kwargs)
        if self.learning_type in [LearningType.Clustering]:
            # Unsupervised learning doesn't need target.
            self.target = target
        else:
            if validate_util.is_non_empty_list(target):
                self.target = target
            else:
                raise Exception("self.get_targets方法不能返回为空，如果是自定义DCStep，请覆盖实现该方法。")

        # 7.4. 获取输出列
        outputs = self.get_outputs(X, y, options, **kwargs)
        if validate_util.is_non_empty_list(outputs):
            self.outputs = outputs
        else:
            raise Exception("self.get_outputs方法不能返回为空，如果是自定义DCStep，请覆盖实现该方法。")

        self.output_cols = [o.name for o in self.outputs]

        # 7.5. 获取是否为二分类任务
        fit_learning_type = kwargs.get("learning_type", None)
        if fit_learning_type is not None:
            if fit_learning_type in LearningType.all:
                self.learning_type = fit_learning_type
                logger.info("设置learning_type: [{}]".format(fit_learning_type))
            else:
                logger.info("指定的learning_type异常，忽略它[{}]。需要是：{}".format(fit_learning_type, LearningType.all))

        if self.learning_type not in LearningType.explicit_types:
            logger.info("当前learning_type不明确[{}]，尝试推理...".format(self.learning_type))
            learning_type = self.inference_learning_type(X, y, options, **kwargs)
            # 只有当返回的learning_type是允许的值时才设置
            if learning_type in LearningType.all:
                self.learning_type = learning_type
                logger.info("推理出:{}".format(learning_type))
            else:
                logger.info("没有推理出来：[{}]".format(learning_type))

        # 7.6. 为二分类任务时，尝试从kwargs中获取正样本及阈值
        if self.learning_type==LearningType.BinaryClassify:
            # 保证binary_threshold为None或[0,1]的小数。
            self.binary_threshold = kwargs.get("binary_threshold", None)
            if not isinstance(self.binary_threshold, float) and self.binary_threshold is not None:
                try:
                    self.binary_threshold = float(self.binary_threshold)
                except ValueError:
                    logger.warn("传入的二分类阈值[{}]不能转为float！将不使用该阈值！".format(self.binary_threshold))
            if isinstance(self.binary_threshold, float):
                if self.binary_threshold<0 or self.binary_threshold>1:
                    logger.warn("传入的二分类阈值[{}]不在[0,1]范围内！将不使用该阈值！".format(self.binary_threshold))
                    self.binary_threshold = None
                self.extra["binary_threshold"] = self.binary_threshold
            # 保证positive_label为None或str
            if not hasattr(self, "positive_label"):
                self.positive_label = None
            positive_label = kwargs.get("positive_label", None)
            if positive_label is not None:
                logger.info(f"fit时设置有positive_label值[{positive_label}],优先使用它，原有值为{self.positive_label}")
                self.positive_label = positive_label
            if self.positive_label is not None:
                self.positive_label = str(self.positive_label)
                self.extra["positive_label"] = self.positive_label

            logger.info("当前为二分类任务，设置正样本[{}]及阈值[{}]".format(self.positive_label, self.binary_threshold))


    def inference_learning_type(self, X, y=None, options=None, **kwargs):
        """推理当前任务的learning_type，处理二分类阈值时引入"""
        pass


    def calc_train_info(self, X):
        """计算训练数据集信息。
        Args:
            X: 输入数据
        """
        pass

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        """预测"""

        pass

    def get_feature_importances(self, columns, feature_importances, data_type):
        """解析模型的特征重要性。

        Args:
            columns: 特征列名
            feature_importances: 特征重要性
            data_type: 特征重要行的数据类型，单机的为float64, 分布式的为double。
        Returns:
            list: list of :class:`dc_model_repo.base.ChartData`
        """
        feature_importance_params = []
        # 1. 包装成参数
        for col_index in range(len(columns)):
            p = Param(columns[col_index], data_type, round(feature_importances[col_index], 4))
            feature_importance_params.append(p)

        # 2. 特征重要性排序
        self.feature_importance = sorted(feature_importance_params, key=lambda _v: _v.value, reverse=True)
        cd = ChartData('featureImportance', 'featureImportance', self.feature_importance)
        return cd

    def get_outputs(self, x, y=None, options=None, **kwargs):
        """计算输出列信息。在模型训练完毕之后调用。

        Returns:
            list: list of :class:`dc_model_repo.base.Output`
        """
        # fixme: 这里为了获取type，其实不需要使用全部的y来构造一个series
        o = Output(name=self.output_cols[0], type=pd.Series(y).dtype.name)
        return [o]

    def get_targets(self, x, y=None, options=None, **kwargs):
        """获取训练目标列的信息, 仅estimator时候需要重写。

        Returns:
            list: list of :class:`dc_model_repo.base.Feature`
        """
        # fixme: 这里为了获取type，其实不需要使用全部的y来构造一个series
        f = Field(name=self.target_cols[0], type=pd.Series(y).dtype.name)
        return [f]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        """获取标签的样本数据。"""

        pass


@six.add_metaclass(abc.ABCMeta)
class BaseUserDefinedDCStep(BaseTransformer):
    """Deprecated. Use :class:`dc_model_repo.step.userdefined_step.UserDefinedTransformer` or :class:`dc_model_repo.step.userdefined_step.UserDefinedEstimator` instead """

    def __init__(self, framework, model_format, input_cols,
                 algorithm_name, source_code_path=None, extension=None, requirements=None, **kwargs):

        super(BaseUserDefinedDCStep, self).__init__(operator=None, framework=framework, model_format=model_format, input_cols=input_cols,
                                                    algorithm_name=algorithm_name, source_code_path=source_code_path, requirements=requirements,
                                                    extension=extension, **kwargs)


@six.add_metaclass(abc.ABCMeta)
class UserDefinedDCStep(BaseUserDefinedDCStep):
    """Deprecated. Use :class:`dc_model_repo.step.userdefined_step.UserDefinedTransformer` instead """

    def __init__(self, input_cols, algorithm_name, framework=FrameworkType.Custom, model_format=None, source_code_path=None, extension=None, requirements=None, **kwargs):
        if source_code_path is None:  #
            from dc_model_repo.base.mr_log import logger
            source_code_path = cls_util.get_source_module(self)
            class_name = cls_util.get_full_class_name(self)
            is_file = os.path.isfile(source_code_path)
            if is_file:
                msg = "它是一个文件，持久化时只有该文件会被保存。"
            else:
                msg = "它是一个目录，持久化时整个目录都会被保存。"
            logger.info("自定义DCStep 类名称：%s，检测到源码路径为：%s，%s" % (class_name, source_code_path, msg))

        super(UserDefinedDCStep, self).__init__(framework=framework, model_format=model_format, input_cols=input_cols, algorithm_name=algorithm_name,
                                                source_code_path=source_code_path, extension=extension, requirements=requirements)

    def delete_py_cache_and_bin_files(self, fs, dir_path):
        from dc_model_repo.base.mr_log import logger
        sub_files = fs.listdir(dir_path)
        for file_name in sub_files:
            file_path = os.path.join(dir_path, file_name)
            if file_name == "__pycache__" and fs.is_dir(file_path):
                #  条件： 名称是__pycache__，并且是个文件夹
                logger.info("删除Python缓存目录: %s" % file_path)
                fs.delete_dir(file_path)
            elif fs.is_file(file_path) and len(file_name) >= 4 and file_name[-4:] == ".pyc":
                # 条件：是个文件，并且后缀是.pyc
                logger.info("删除Python pyc文件: %s" % file_path)
                fs.delete_file(file_path)
            else:
                if fs.is_dir(file_path):
                    # 如果是目录，并且不是__pycache__ 就递归删除
                    self.delete_py_cache_and_bin_files(fs, file_path)

    def get_params(self):
        return None

    def prepare(self, step_path, **kwargs):
        self.load_model(step_path)

    def persist_model(self, fs, destination):
        pass


@six.add_metaclass(abc.ABCMeta)
class UserDefinedDCEstimator(UserDefinedDCStep, BaseEstimator):
    """Deprecated. Use :class:`dc_model_repo.step.userdefined_step.UserDefinedEstimator` instead """

    def __init__(self, input_cols,
                 algorithm_name,
                 target_cols=['label'],
                 output_cols=['prediction'],
                 framework=FrameworkType.Custom,
                 model_format=None,
                 source_code_path=None,
                 extension=None,
                 requirements=None,
                 **kwargs):
        UserDefinedDCStep.__init__(self,
                                   input_cols=input_cols,
                                   algorithm_name=algorithm_name,
                                   framework=framework,
                                   model_format=model_format,
                                   source_code_path=source_code_path,
                                   extension=extension,
                                   requirements=requirements,
                                   **kwargs)

        BaseEstimator.__init__(self, output_cols=output_cols, target_cols=target_cols)

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)


# 该类已弃用，所有的Step都需要额外的模型, 请使用FileSystemDCStep。
class ModelWrapperDCStep(DCStep):
    """Deprecated."""

    pass

# from dc_model_repo.step.aps_step import PipelineInitDCStep as RealPipelineSampleDataCollectDCStep
#
#
# class PipelineInitDCStep(RealPipelineSampleDataCollectDCStep):
#     def __init__(self, **kwargs):
#         from dc_model_repo.base.mr_log import logger
#         logger.warning("此类已经放到dc_model_repo.step.aps_step.PipelineSampleDataCollectDCStep中维护，请尽快使用"
#                     "from dc_model_repo.step.aps_step import PipelineInitDCStep 替代。" )
#         super(PipelineInitDCStep, self).__init__(**kwargs)
