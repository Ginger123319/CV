# -*- encoding: utf-8 -*-
import os
import collections
import json
import pandas as pd
import uuid
from pyspark.ml import PipelineModel
from pyspark.sql.types import DoubleType
from sklearn.pipeline import Pipeline

from dc_model_repo import __version__
from dc_model_repo import model_repo_client
from dc_model_repo.base import BaseOperator, Field, ChartData, DictSerializable, TrainInfo, LearningType
from dc_model_repo.base import Module, PipelineModelEntry, Output, Param
from dc_model_repo.base import FrameworkType
from dc_model_repo.base import DatasetType
from dc_model_repo.base import file_system
from dc_model_repo.base import collector_manager
from dc_model_repo.base.file_system import LocalFileSystem
from dc_model_repo.step.aps_step import FakeUnSerializableDCStep
from dc_model_repo.step.base import DCStep, BaseEstimator
from dc_model_repo.step.pipeline_transformer_step import PipelineTransformerStep
from dc_model_repo.step.sklearn_step import SKLearnDCEstimator, SKLearnDCTuningEstimator
from dc_model_repo.step.spark_step import SparkDCEstimator, SparkDCTuningEstimator
from dc_model_repo.step.java_model_step import GeneratedJavaEstimator, FakeSparkStep
from dc_model_repo.util import validate_util, dataset_util, spark_util, time_util, cls_util, json_util, pkl_util, zip_util, str_util, pmml_util


class PipelineMetaData(DictSerializable):
    """使用这个类来封装Pipeline相关信息，pipeline模型文件中meta.json是从这个类的实例序列化来的

    Args:
        id (str): pipeline id 每次生成pipeline都会生成一个uuid
        name (str): pipeline名称
        timestamp (int): 时间戳（毫秒级）
        version (str): dc_model_repo的版本
        module_id (str): 工作流中生成pipeline所在算子的模块id
        steps_meta (dict): Pipeline持久化后各个step的id和相对路径
        modules (list): 工作流中各个step的模块信息
        class_name (str): Pipeline的类名
        language (str): six包中python语言版本
        framework (str): 框架类型。Refer to :class:`dc_model_repo.base.FrameworkType`
        learning_type (str): 学习类型。 Refer to: :class:`dc_model_repo.base.LearningType`
        algorithm_name (str): Steps中最后一个Estimator的算法名称
        default_metric (str): 默认metric,用于产品页面展示默认评估结果。 Refer to: "attr":`mapping`
        estimator_train_info (dict): Estimator训练时间、数据行列等信息。
        pipeline_model (list): 将可合并的 step 合并成的 pipeline 信息。
        graph (list): 工作流的运行图
        input_features (list): 进入 pipeline 的输入列信息。 list of :class:`dc_model_repo.base.Field`
        input_type (str): 输入 pipeline 的数据的类型
        estimator_params (list): Estimator 参数。 list of :class:`dc_model_repo.base.Param`
        target (list): Estimator 的目标列。 list of :class:`dc_model_repo.base.Field`
        outputs (list): Estimator 的输出列。 list of :class:`dc_model_repo.base.Field`
        description (str): 模型描述
        extension (dict): 扩展信息，包括构造 pipeline 所在算子所处的环境信息。
        attachments (list): 模型附件
    """

    def __init__(self, id, name, timestamp, version, module_id, steps_meta, modules, class_name, language, framework, learning_type,
                 algorithm_name, default_metric, estimator_train_info, pipeline_model, graph, input_features, input_type,
                 estimator_params, target, outputs, description, extension, attachments):
        self.id = id
        self.name = name
        self.timestamp = timestamp
        self.version = version
        self.module_id = module_id
        self.steps_meta = steps_meta
        self.modules = modules
        self.class_name = class_name
        self.language = language
        self.framework = framework
        self.learning_type = learning_type
        self.algorithm_name = algorithm_name
        self.default_metric = default_metric
        self.estimator_train_info = estimator_train_info
        self.pipeline_model = pipeline_model
        self.graph = graph
        self.input_features = input_features
        self.input_type = input_type
        # self.labels = labels  # 字符串标签列
        # self.sample_data_path = sample_data_path # 原因: sample_data_path不是Pipeline的元信息，从meta中分离出来。
        # self.target_sample_data_path = target_sample_data_path
        self.estimator_params = estimator_params
        self.target = target
        self.outputs = outputs
        self.description = description
        # self.proba_feature = proba_feature
        self.extension = extension
        self.attachments = attachments

    @classmethod
    def field_mapping(cls):
        return {
            'id': 'id',
            'module_id': 'moduleId',
            'steps_meta': 'stepsMeta',
            'name': 'name',
            'timestamp': 'timestamp',
            'version': 'version',
            'class_name': 'className',
            'language': 'language',
            'framework': 'framework',
            'learning_type': 'learningType',
            'algorithm_name': 'algorithmName',
            'default_metric': 'defaultMetric',
            'estimator_train_info': 'estimatorTrainInfo',
            'graph': 'graph',
            'input_type': 'inputType',
            # 'labels': 'labels',
            # 'sample_data_path': 'sampleDataPath',
            # 'target_sample_data_path': 'targetSampleDataPath',
            'description': 'description',
            'attachments': 'attachments',
            'extension': 'extension'
        }

    def to_dict(self):
        result_dict = self.member2dict(self.field_mapping())
        result_dict['pipelineModel'] = DictSerializable.to_dict_if_not_none(self.pipeline_model)
        result_dict['inputFeatures'] = DictSerializable.to_dict_if_not_none(self.input_features)
        result_dict['estimatorParams'] = DictSerializable.to_dict_if_not_none(self.estimator_params)
        result_dict['estimatorTrainInfo'] = DictSerializable.to_dict_if_not_none(self.estimator_train_info)
        result_dict['target'] = DictSerializable.to_dict_if_not_none(self.target)
        result_dict['outputs'] = DictSerializable.to_dict_if_not_none(self.outputs)
        result_dict['modules'] = DictSerializable.to_dict_if_not_none(self.modules)
        # result_dict['probaFeature'] = None if self.proba_feature is None else [ep.to_dict() for ep in self.proba_feature]
        return result_dict

    @staticmethod
    def len_construction_args():
        return 24

    @classmethod
    def load_from_dict(cls, dict_data):
        # 1. 填充简单Key
        bean = super(PipelineMetaData, cls).load_from_dict(dict_data)

        # 2. 设置复杂key
        bean.pipeline_model = PipelineModelEntry.load_from_dict_list(dict_data.get('pipelineModel'))
        bean.input_features = Field.load_from_dict_list(dict_data['inputFeatures'])

        bean.estimator_params = Param.load_from_dict_list(dict_data.get('estimatorParams'))
        bean.estimator_train_info = TrainInfo.load_from_dict(dict_data.get('estimatorTrainInfo'))
        bean.target = Field.load_from_dict_list(dict_data['target'])
        bean.outputs = Output.load_from_dict_list(dict_data['outputs'])
        bean.modules = Module.load_from_dict_list(dict_data.get('modules'))
        # bean.proba_feature = Feature.load_from_dict_list(dict_data['probaFeature'])

        return bean


class DCPipeline(BaseOperator):
    """APS对Pipeline模型的抽象。 带有一个或者多个transformer和1个estimator类型的DCStep.

    Args:
        steps (list): DCStep数组， 不能为空。
        name (str): 模型名称，可选，默认为 `pipeline_{pipeline.id}`。
        learning_type (str): 模型类型，Refer to :class:`dc_model_repo.base.LearningType`。
        input_type(str): 输入数据的类型。
        input_features (list): 进入 pipeline 的输入列信息。 list of :class:`dc_model_repo.base.Field`。
        sample_data: 样本数据（剔除目标列）。一般取 pipeline 初始化时候的样本数据。
        target_sample_data: 目标列样本数据。
        attachments (list): 模型附件, 字典数组。
        performance (list): 模型性能。 list of :class:`dc_model_repo.base.ChartData`
        modules (list): 工作拓朴中模块信息。
        description (str): 模型描述。
        extension (dict): 扩展字段。
        skip_validation(bool): 是否在predict时跳过输入数据校验。
            仅用于分布式模型转单机化执行时。默认False
        performance_file: 模型评估附件地址，字符串类型文件路径。
            路径可以是名为performance.json的文件或者包含附件的文件夹路径。
            当performance不为None时，不再对performance.json文件进行解析。
        pipeline_id: 模型ID
        load_path: 如果Pipeline是通过load加载的，这里为模型路径。如果为None，表示Pipeline是新创建的。
        **kwargs: 备用
    """

    _PMML_FILE_TYPE = "pmml"

    def __init__(self, steps, name, learning_type, input_type, input_features, sample_data=None,
                 target_sample_data=None, attachments=None, performance=None, modules=None,
                 description=None, extension=None, skip_validation=None, performance_file=None,
                 pipeline_id=None, load_path=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        print("pipeid is {}".format(pipeline_id))
        if pipeline_id is None:
            pipeline_id = str(uuid.uuid4())
            logger.info("自动生成PipelineID：{}".format(pipeline_id))
        else:
            pipeline_id = str(pipeline_id)
            logger.info("明确指定了PipelineID：{}".format(pipeline_id))
        BaseOperator.__init__(self, pipeline_id, extension)
        if not validate_util.is_non_empty_list(steps):
            steps = []

        for s in steps:
            if isinstance(s, FakeUnSerializableDCStep):
                logger.warning("当前DCPipeline对象无法对数据进行评估（调用predict方法），因为有无法反序列化的DCStep, %s" % str(s))

        self.steps = steps
        self.name = name
        self.step_path_mapping = {}
        self.learning_type = learning_type
        self.input_features = input_features
        self.sample_data = sample_data
        self.algorithm_name = None
        self.input_type = input_type
        self.default_metric = LearningType.mapping.get(learning_type)
        self.pipeline_model = []

        self.attachments = attachments
        self.graph = None
        self.modules = None
        self.performance = performance
        self.performance_file = None
        if performance_file is not None:
            self.performance_file = str(performance_file)

        self.estimator_train_info = None

        self.step_path_mapping = None
        self.target = None  # Pipeline的标签列字段，将从Estimator中推断。
        self.outputs = None  # Pipeline的输出字段
        self.target_sample_data = target_sample_data
        self.estimator_params = None
        self.description = description
        self.skip_validation = False if skip_validation is None else skip_validation
        self.load_path = load_path

        # 1. 参数校验
        if self.input_features is None:
            raise Exception("参数input_features不能为空。")

        if self.input_type is None:
            raise Exception("参数input_type不能为空。")

        # 2. 设置评估器参数, 和target以及target样本数据，找到最后一个estimator, 使用其参数
        estimator_step = self.get_estimator_step()
        if estimator_step is not None:
            self.estimator_params = estimator_step.params
            # 2.1. 设置target
            self.target = estimator_step.target

            # 2.2. 设置输出列
            self.outputs = estimator_step.outputs

            # 2.3. 设置算法名称
            self.algorithm_name = estimator_step.algorithm_name

            # 2.4. 设置训练信息
            self.estimator_train_info = estimator_step.train_info

            # 2.5. 设置特征重要性信息。
            if self.performance is None:
                self.performance = []

            # 2.6. 设置标签列样本数据
            if self.target_sample_data is None:
                # logger.warning("DCPipeline模型的标签列样本数据为空，尝试使用estimator的标签列样本数据。")
                self.target_sample_data = estimator_step.target_sample_data

            # not_has_feature_importance_in_performance = True
            # for cd in self.performance:
            #     if cd.type == 'featureImportance':
            #         not_has_feature_importance_in_performance = False
            #
            # if not not_has_feature_importance_in_performance:
            #     logger.info("传入的performance参数中已经包含特征重要性。")
            #
            # feature_importance = estimator_step.feature_importance
            # if feature_importance is not None:
            #     cd = ChartData('featureImportance', 'featureImportance', feature_importance)
            #     logger.info("解析到评估器含有特征重要性数据。")
            # self..append(cd)
        else:
            logger.warning("当前Pipeline中没有Estimator.")

        # 3. 设置模块信息
        if validate_util.is_non_empty_list(modules):
            # 3.1. 创建查询索引
            modules_mapping = self.create_modules_mapping(modules)

            # 3.2. 对step进行groupBy
            sorted_modules = []
            for s in steps:
                module_id = s.module_id
                m = modules_mapping.get(module_id)
                if m is None:
                    raise Exception("Step id=%s, 算法名称=%s, 没有找到模块信息，模块id=%s" %
                                    (s.id, s.algorithm_name, module_id))
                m.steps.append({'id': s.id, 'algorithmName': s.algorithm_name})
                if validate_util.is_non_empty_list(sorted_modules):
                    latest_module = sorted_modules[-1]
                    if latest_module.id != m.id:
                        sorted_modules.append(m)
                else:
                    sorted_modules.append(m)

            # 3.3. 设置pipeline的module
            self.modules = sorted_modules
        else:
            # logger.warning("没有设置模块信息，将使用%s作为默认模块名称。" % DCPipeline.FILE_MODULES)
            pass

        logger.info("End flag of DCPipeline __init__.")

    FILE_MODULES = 'modules'  # Note: (取消模块概念后废弃)
    FILE_STEPS = 'steps'
    FILE_DEFAULT_MODULE = 'default'

    def set_steps_paths(self, step_path_mapping):
        self.step_path_mapping = step_path_mapping

    def get_estimator_step(self):
        s_len = len(self.steps)
        for i in range(s_len):
            s = self.steps[s_len - i - 1]
            # 继承了BaseEstimator是 DCEstimator
            if isinstance(s, BaseEstimator):
                return s
        return None

    @staticmethod
    def create_modules_mapping(modules):
        modules_mapping = {}
        for m in modules:
            modules_mapping[m.id] = m
        return modules_mapping

    def predict(self, X, **kwargs):
        """进行预测。

        Args:
            X: 可以是pandas或者PySpark的DataFrame或np.ndarray或dict，由`input_type`决定。
            **kwargs: 向后传递的额外参数。
                - calc_all_proba： 分类时可用，是否计算概率列。默认为True

        Returns:
        """

        # 1. transform
        X = self.predict_transformers(X)

        # 2. predict
        return self.predict_estimator(X, **kwargs)

    def predict_estimator(self, X, **kwargs):
        """进行最后一个Estimator的预测。

        Args:
            X: 前置transformer的输出。
            **kwargs: 向后传递的额外参数。
                - calc_all_proba： 分类时可用，是否计算概率列。默认为True

        Returns:
        """
        from dc_model_repo.base.mr_log import logger
        estimator = self.steps[-1]
        # 如果是分类任务，且没有设置计算概率列的标识，将填上默认值True
        if "calc_all_proba" not in kwargs:
            from dc_model_repo.base import LearningType
            if self.learning_type in [LearningType.BinaryClassify, LearningType.MultiClassify]:
                logger.info("当前模型学习类型为[{}], 但在predict时没有设置calc_all_proba参数，将填上默认值True".format(self.learning_type))
                kwargs["calc_all_proba"] = True

        result = estimator.predict(X, **kwargs)
        if isinstance(result, pd.DataFrame):
            return result
        else:
            import dask.dataframe as dd
            if isinstance(result, dd.DataFrame):
                result_df = result.compute()
                return result_df
            else:
                logger.info("注意：当前Pipeline的预测结果既不是pandas的DataFrame，又不是dask的DataFrame： {}".format(type(result)))
                return result

    def predict_transformers(self, X):
        """校验输入的数据，并执行所有transformer的预测。"""
        from dc_model_repo.base.mr_log import logger

        # 1. 对输入的数据进行类型检查
        if self.skip_validation:
            logger.info("当前Pipeline跳过了输入数据校验环节！")
        else:
            if isinstance(X, pd.DataFrame):
                logger.info("输入数据特征:\n%s" % str(X.dtypes))
            if self.input_type == DatasetType.ArrayData:
                self.input_type = DatasetType.Dict
            X = dataset_util.validate_and_cast_input_data(X, self.input_type, self.input_features, remove_unnecessary_cols=True)
            if isinstance(X, pd.DataFrame):
                logger.info("类型转换后输入数据特征:\n%s" % str(X.dtypes))

        for i, s in enumerate(self.steps[:-1]):
            # logger.info("Step %s 将要转换DF, 格式为: \n%s\"" % (str(s), X.dtypes)) # Note 此日志影响性能
            try:
                X = s.transform(X)
            except Exception as e:
                logger.error("调用Step=%s, 在Pipeline 为位于第%d个(从1计数的)Step时出错。" % (str(s), i + 1))
                raise e
        return X

    def serialize_steps_path(self, destination):
        return os.path.join(destination, DCPipeline.FILE_STEPS)

    def serialize_performance_path(self, destination):
        return destination + "/performance/performance"

    def serialize_pmml_path(self, destination):
        return self.serialize_data_path(destination) + "/pipeline.pmml"

    def serialize_labels_file_path(self, destination):
        return self.serialize_data_path(destination) + "/labels.json"

    def serialize_sklearn_pipeline_path(self, destination):
        return self.serialize_data_path(destination) + "/sklearnPipeline.pkl"

    def serialize_performance_meta_path(self, destination):
        return self.serialize_performance_path(destination) + "/performance.json"

    def persist_prepare(self, fs, destination):
        fs.make_dirs(self.serialize_data_path(destination))

    def persist_steps(self, fs, step_serialize_path, copy_steps, persist_sample_data, persist_explanation):
        from dc_model_repo.base.mr_log import logger
        # 1.创建模块映射
        # (取消模块概念后废弃)
        # modules_mapping = None
        # if self.modules is not None:
        #     modules_mapping = self.create_modules_mapping(self.modules)

        # 2. 计算Step在Pipeline中的路径
        def calc_step_dir(step, index):
            # Note: (取消模块概念后废弃)
            # if modules_mapping is not None:
            #     m = modules_mapping.get(step.module_id)
            #     module_name = m.name
            # else:
            #     module_name = DCPipeline.FILE_DEFAULT_MODULE
            return "%s_%s" % (index, step.algorithm_name)

        # 3. 分别持久化Step
        len_steps = len(self.steps)

        # 3.1. 计算出step个数的位数
        len_num_steps = len(str(len_steps))
        steps_meta = []
        for index in range(len_steps):
            step = self.steps[index]
            step_dir = calc_step_dir(step, str(index).zfill(len_num_steps))
            steps_meta.append({"id": step.id, "dirName": step_dir})
            dest = os.path.join(step_serialize_path, step_dir)

            logger.info("Step id=%s序列化的地址：%s" % (step.id, dest))
            if copy_steps:
                logger.info("开始复制Step id=%s。" % step.id)
                if self.step_path_mapping is not None:
                    p = self.step_path_mapping.get(step)
                    if p is None:
                        raise Exception('没有找到Step"%s" 所在的地址。' % str(step))
                    logger.info("Step源地址:%s" % p)
                    fs.copy(p, dest)
                else:
                    raise Exception('请调用"set_steps_paths(step_path_mapping)" 设置Step所在的路径。')
            else:
                try:
                    step.persist(dest, fs_type=fs.get_name(), persist_sample_data=persist_sample_data, persist_explanation=persist_explanation)
                except Exception as e:
                    logger.error("持久化Step=%s失败。" % str(step))
                    raise e
        return steps_meta

    def persist(self, destination=None, fs_type=None, copy_steps=False, generate_pmml=True, generate_sklearn_pipeline=True, persist_sample_data=True,
                persist_explanation=True, persist_sample_data_each_step=False, **kwargs):
        """序列化Pipeline到文件系统。

        Args:
            destination (str): 持久化地址。
            fs_type (str): 文件系统类型。
            copy_steps (bool): 是否复制而不是重新序列化step， 如果设置为True时要保证 `step_path_mapping` 正确设置。
            generate_pmml (bool): 是否尝试生成PMML文件，默认为True。
            persist_sample_data: 是否序列化样本数据。
            generate_sklearn_pipeline: 是否生成SKLearn Pipeline。
            persist_explanation: 是否序列化模型可视化数据。
            persist_sample_data_each_step: 是否持久化每个step的样本数据，默认False，不持久化。
            **kwargs: 备用

        """
        from dc_model_repo.base.mr_log import logger

        # 1. 校验是否有 FakeUnSerializableDCStep
        for s in self.steps:
            if isinstance(s, FakeUnSerializableDCStep):
                if not copy_steps:
                    raise Exception("当前steps中FakeUnSerializableDCStep，它必须是已经序列化过的，需要配合copy_step=True使用。")

        if destination is None:
            extension_collector = collector_manager.get_extension_collector()
            logger.info("Type of extension_collector: [{}]".format(type(extension_collector)))
            if extension_collector is not None:
                data_root = extension_collector.get_data_path()
                logger.info(("Get data_root from extension_collector: {}".format(data_root)))
            else:
                data_root = "."
            destination = data_root + "/pipeline"

        if fs_type is None:
            fs_type = self.get_fs_type()

        # 2. Pipeline写入的文件系统和最后一个Step保持一致。
        logger.info('准备持久化Pipeline "%s", id=%s 到: %s' % (str(self), self.id, destination))
        fs = file_system.instance_by_name(fs_type)

        # 3. 创建目录
        self.persist_prepare(fs, destination)

        # 4. 持久化 steps
        steps_meta = self.persist_steps(fs, self.serialize_steps_path(destination), copy_steps, persist_sample_data_each_step, persist_explanation)

        # 5. 判定pipeline 使用的框架
        frameworks = list(set([t.framework for t in self.steps]))
        _len_frameworks = len(frameworks)
        # 当只有一种框架类型，且该类型不为Custom时候，才将Pipeline的框架类型设置成当前值
        if _len_frameworks == 1 and frameworks[0] != FrameworkType.Custom:
            framework = frameworks[0]
        elif _len_frameworks == 0:
            framework = None
        else:
            framework = FrameworkType.Mixed

        # 6. 构建Pipeline本身的运行图
        run_graph_connections = None
        if len(self.steps) > 0:
            first_step = self.steps[0]
            first_step_connection = StepConnection(source_step_id=None, source_io_name='input_data',
                                                   target_io_name='output_data', target_step_id=first_step.id).to_dict()

            run_graph_connections = [first_step_connection]
            for index in range(len(self.steps) - 1):
                p_step = self.steps[index]
                n_step = self.steps[index + 1]

                sc = StepConnection(source_step_id=p_step.id, source_io_name='input_data',
                                    target_io_name='output_data', target_step_id=n_step.id).to_dict()

                run_graph_connections.append(sc)

        # 7. 尝试生成PMML，如果可以构建上线时的运行图 TODO k 尝试下pmml格式的模型逻辑放到各框架的pipeline中实现，带来的问题：这样要引入新的子pipeline
        if generate_pmml is True:
            build_pmml_result = self.build_pmml()
            if build_pmml_result is not None:
                # 生成pmml 文件
                pmml_file_path = self.serialize_pmml_path(destination)
                fs.write_bytes(pmml_file_path, build_pmml_result.pmml_bytes)
                logger.info("写入PMML文件到: %s" % pmml_file_path)
                # 记录PMML中包涵的step
                step_ids = [s.id for s in build_pmml_result.include_steps]
                p_entry = PipelineModelEntry(file_name="pipeline.pmml", file_type=DCPipeline._PMML_FILE_TYPE, contains_steps=step_ids)
                self.pipeline_model.append(p_entry)
            else:
                logger.warning("构建得到的PMML为空。")
        else:
            logger.info("已设置忽略构建PMML。")

        # 8. 落地labels 文件作为pmml的补丁 TODO k 这一块跟前面的生成pmml的pipeline可以放到一起
        dc_estimator = self.get_estimator_step()

        if dc_estimator is not None and isinstance(dc_estimator, (SparkDCEstimator, SparkDCTuningEstimator)):
            labels = dc_estimator.labels

            if validate_util.is_non_empty_list(labels):
                logger.info("检测到有Spark字符串label训练，持久化labels。")
                fs.write_bytes(self.serialize_labels_file_path(destination),
                               str_util.to_bytes(json_util.to_json_str(labels)))

        # 9. 尝试合并SKLearn算子
        if generate_sklearn_pipeline:
            if len(self.steps) > 0:
                sklearn_build_result = self.build_sklearn_pipeline()
                if sklearn_build_result is not None:
                    # 写入PKL文件
                    fs.write_bytes(self.serialize_sklearn_pipeline_path(destination),
                                   pkl_util.serialize2bytes(sklearn_build_result.sklearn_pipeline))

                    # 记录SKlearn Pipeline 中包涵的 step
                    step_ids = [s.id for s in sklearn_build_result.include_steps]
                    p_entry = PipelineModelEntry(file_name="sklearnPipeline.pkl", file_type="pkl", contains_steps=step_ids)
                    self.pipeline_model.append(p_entry)

        # 10. 落地评估信息
        # 先处理performance_file.
        # 路径可以是名为performance.json的文件或者包含附件的文件夹路径。当performance不为None时，不再对performance.json文件进行解析。
        logger.info("开始处理performance内容...")
        if self.performance_file is not None and fs.exists(self.performance_file):
            performance_json_file = None
            files_copied = []
            if fs.is_file(self.performance_file):
                if self.performance_file.endswith(".json"):
                    performance_json_file = self.performance_file
            else:
                p_dir = self.serialize_performance_path(destination)
                if not fs.exists(p_dir):
                    fs.make_dirs(self.serialize_performance_path(destination))
                for v in os.listdir(self.performance_file):
                    vv = os.path.join(self.performance_file, v)
                    if fs.is_file(vv):
                        if v == "performance.json":
                            performance_json_file = vv
                        else:
                            target_p = os.path.join(p_dir, v)
                            logger.info("拷贝文件... {} -> {}".format(vv, target_p))
                            fs.copy(vv, target_p)
                            files_copied.append(v)
                    else:
                        logger.info("跳过文件夹: {}".format(vv))
            if performance_json_file is not None:
                if validate_util.is_empty_list(self.performance):
                    try:
                        json_str = fs.read_bytes(performance_json_file).decode("utf-8")
                        p_data = json.loads(json_str)
                        for d in p_data:
                            cd = ChartData(d.get('name'), d.get('type'), d.get('data'))
                            self.performance.append(cd)
                    except Exception as e:
                        logger.warn("Can't decode file [{}] with utf-8: ERROR: \n{}".format(performance_json_file, repr(e)))
                else:
                    logger.info("入参明确指定了performance，现在忽略[{}]".format(performance_json_file))
            if len(files_copied) > 0:
                for f in files_copied:
                    cd = ChartData(name=f, type='file', data='performance/{}'.format(f))
                    self.performance.append(cd)
        else:
            if self.performance_file is None:
                logger.info("performance_file为None，跳过文件处理。")
            else:
                error_msg = "指定的performance_file文件不存在: [{}]".format(self.performance_file)
                logger.error(error_msg)
                raise Exception(error_msg)

        if validate_util.is_non_empty_list(self.performance):
            fs.make_dirs(self.serialize_performance_path(destination))
            performance_str = str_util.to_bytes(json_util.to_json_str([e.to_dict() for e in self.performance]))
            fs.write_bytes(self.serialize_performance_meta_path(destination), performance_str)
        else:
            logger.warn("请注意：没有performance数据！")

        logger.info("结束处理performance内容。")

        # 11. 生成Meta对象
        pipeline_meta = None
        if self.load_path is not None:
            try:
                meta_path = str(self.load_path) + '/meta.json'
                logger.info("Loading meta info from: {}".format(meta_path))
                pipeline_meta = PipelineMetaData.load_from_json_str(str_util.to_str(fs.read_bytes(meta_path)))
                logger.info("Meta info loaded.")
            except Exception as e:
                logger.info("Failed to load meta info: {}".format(e))

        if pipeline_meta is None:
            pipeline_meta = PipelineMetaData(id=self.id, name=self.name,
                                             timestamp=time_util.current_time_millis(),
                                             version=__version__, module_id=self.module_id,
                                             steps_meta=steps_meta,
                                             modules=self.modules,
                                             class_name=cls_util.get_full_class_name(self),
                                             framework=framework, learning_type=self.learning_type,
                                             algorithm_name=self.algorithm_name,
                                             default_metric=self.default_metric,
                                             estimator_train_info=self.estimator_train_info,
                                             pipeline_model=self.pipeline_model,
                                             graph=run_graph_connections,
                                             input_features=self.input_features,
                                             input_type=self.input_type,
                                             estimator_params=self.estimator_params,
                                             target=self.target,
                                             outputs=self.outputs,
                                             extension=self.extension,
                                             attachments=self.attachments,
                                             language=cls_util.language(),
                                             description=self.description)

        # 12. 生成pipeline的meta文件
        json_str = pipeline_meta.to_json_string()
        logger.info("Meta info: {}".format(json_str))
        meta_path = self.serialize_meta_path(destination)

        fs.write_bytes(meta_path, str_util.to_bytes(json_str))

        # 13. 落地样本数据
        if self.load_path is not None and fs.exists(self.load_path):
            origin_sample = self.serialize_sample_data_path(self.load_path)
            origin_target_sample = self.serialize_target_sample_data_path(self.load_path)
            if fs.exists(origin_sample):
                logger.info("Copy sample data from: {}".format(origin_sample))
                fs.write_bytes(self.serialize_sample_data_path(destination), fs.read_bytes(origin_sample))
            if fs.exists(origin_target_sample):
                logger.info("Copy sample target data from: {}".format(origin_target_sample))
                fs.write_bytes(self.serialize_target_sample_data_path(destination), fs.read_bytes(origin_target_sample))
        else:
            self.persist_sample_data(fs, destination, persist_sample_data)

        logger.info("End flag of persist method.")

    def prepare(self):
        """加载各个step"""

        # 1. 分步加载Step
        for s in self.steps:
            step_path = self.step_path_mapping.get(s)
            s.prepare(step_path)

    def build_sklearn_pipeline(self):
        from dc_model_repo.base.mr_log import logger
        logger.info("开始尝试构建SKLearn Pipeline. ")
        # 1. 判断最后一个Step是否为Estimator: sklean pipeline 否则调用 predict报错。

        if not isinstance(self.steps[-1], (SKLearnDCEstimator, SKLearnDCTuningEstimator)):
            logger.info("Pipeline的最后一个Step不是SKLearnDCEstimator，结束构建。")
            return None

        # 2. 查找可以合并的 Transformer, 并记录下来
        pipeline_steps = self.trace_step_by_framework(FrameworkType.SKLearn)
        logger.info("要合并的step为：")
        for s in pipeline_steps:
            logger.info("{} ->\n {}".format(s.id, s.model))

        # 3. 生成Pipeline对象。

        sklearn_pipeline = Pipeline(steps=[(s.algorithm_name + "_" + s.id, s.model) for s in pipeline_steps])
        logger.info("构建得到SKLearn Pipeline为: %s" % sklearn_pipeline)
        return BuildSKLearnPipelineResult(sklearn_pipeline, pipeline_steps)

    def trace_step_by_framework(self, framework):
        """
        对self.steps进行反向遍历，获取framework一致的连续step

        Args:
            framework: 要查找的framework，类型见：dc_model_repo.base.FrameworkType

        Returns: 满足条件的step列表

        """
        pipeline_steps = []
        step_len = len(self.steps)
        for i in range(step_len)[::-1]:
            s = self.steps[i]
            if hasattr(s, "framework") and getattr(s, "framework") == framework:
                pipeline_steps.append(s)
            else:
                break

        pipeline_steps.reverse()
        return pipeline_steps

    def trace_step_by_type(self, cls):
        """从最后开始查找连续类型的Step。 Deprecated. Use trace_step_by_framework instead.


        Args:
            cls: 类

        Returns: 连续的同类的step

        """
        import warnings
        warnings.warn("This method is deprecated. Use trace_step_by_framework instead.", DeprecationWarning)

        pipeline_steps = []
        step_len = len(self.steps)
        for i in range(step_len):
            reserve_i = step_len - i - 1
            s = self.steps[reserve_i]
            if isinstance(s, cls):
                pipeline_steps.append(s)
            else:
                break

        pipeline_steps.reverse()
        return pipeline_steps

    def build_pmml(self):
        from dc_model_repo.base.mr_log import logger
        logger.info("开始尝试构建PMML文件。")

        # 1. 判断最后一个Step是否为Estimator TODO k 这个方法只适用于基于Spark的模型，把它放到SparkEstimator中

        if not isinstance(self.steps[-1], (SparkDCEstimator, SparkDCTuningEstimator)):
            logger.info("Pipeline的最后一个Step不是SparkDCEstimator或SparkDCTuningEstimator，结束构建。")
            return None

        # 2. 查找可以合并的 Transformer, 并记录下来
        pipeline_steps = self.trace_step_by_framework(FrameworkType.Spark)

        # 3. 构建成PipelineModel

        # model 有可能是pipeline
        if len(pipeline_steps) == 1 and isinstance(pipeline_steps[0].model, PipelineModel):
            pipeline_model = pipeline_steps[0].model
            pipeline_stages = pipeline_model.stages
        else:
            pipeline_stages = []

            for s in pipeline_steps:
                if isinstance(s, PipelineTransformerStep):
                    sub_pipeline_model = s.model
                    pipeline_stages.extend(sub_pipeline_model.stages)
                else:
                    pipeline_stages.append(s.model)
            pipeline_model = PipelineModel(stages=pipeline_stages)
        logger.info("构建PMML文件，生成的PipelineModel: %s" % str(pipeline_stages))

        # 4. 生成DataFrame()
        schema = pipeline_steps[0].spark_df_schema
        label_col = self.get_estimator_step().target_cols[0]
        fields = schema.fields
        for f in fields:  # label 的类型要修正为double
            if f.name == label_col:
                f.dataType = DoubleType()
                break

        # 5. 对input_df和pipeline checkout(适用于工作流和非工作流模式)
        spark = spark_util.get_spark_session()
        input_df = spark.createDataFrame([], schema=schema, verifySchema=False)

        # build_pmml_cache_persist_dir = "/tmp/build_pmml/%s" % uuid.uuid4()
        # logger.info("构建PMML缓存目录：%s" % build_pmml_cache_persist_dir)
        # pipeline_model.save("%s/model" % build_pmml_cache_persist_dir)  # 目录不存在会自动创建
        # input_df.write.parquet("%s/input_df.parquet" % build_pmml_cache_persist_dir)

        # 6. 判断Spark版本使用对应的JPMML的API生成PMML文件

        pmml_bytes = pmml_util.convert2pmml(sc=spark.sparkContext, input_df=input_df, pipeline_model=pipeline_model)

        logger.info("构建PMML成功。")
        return BuildPMMLResult(pmml_bytes, include_steps=pipeline_steps)

    @staticmethod
    def convert_graph_to_list(graph):
        # 1. 生成 parent->child 的映射图
        graph_mapping = {}
        for connection in graph:
            sc = StepConnection.load_from_dict(connection)
            if sc.source_step_id is None:
                if graph_mapping.get('root') is not None:
                    raise Exception("暂时不支持运行图中有两个顶级")
                graph_mapping['root'] = sc.target_step_id
            else:
                graph_mapping[sc.source_step_id] = sc.target_step_id

        # 2. 从上往下找得到列表
        block_id_first = graph_mapping.get('root')
        if block_id_first is None:
            raise Exception("运行图中没找到根节点，请查看运行图是否正确。")
        block_id_ordered_list = [block_id_first]
        block_id_tmp = block_id_first
        while graph_mapping.get(block_id_tmp) is not None:
            _b = graph_mapping.get(block_id_tmp)
            block_id_ordered_list.append(_b)
            block_id_tmp = _b

        return block_id_ordered_list

    @staticmethod
    def load(path, fs_type=None, model_repo_server_portal=None, jar_path=None, debug_log=False, download_timeout=None):
        """加载模型。

        Args:
            path: 支持本地文件目录或者"model://"格式的url
            fs_type: 文件系统类型
            model_repo_server_portal: MR 服务地址，比如 ``http://dev.aps.zetyun.cn``
            jar_path: 将spark分布式模型在单机加载时，要用到java服务，这时可以用该参数指定包含这个服务的jar包路径。
                非分布式转单机时不需要设置。且当这个jar包已经在classpath上时也不需要设置。
            debug_log: 将spark分布式模型在单机加载时，设置java服务的日志级别，True对应debug级别，False对应info级别。
            download_timeout: tuple格式 (connect timeout, read timeout)，
                当path为url时，下载模型的超时时间，以秒为单位，默认为(30, 60)

        Returns:
            DCPipeline: 加载过的pipeline,使用之前需要先调用其方法 :meth:`prepare`

        """
        from dc_model_repo.base.mr_log import logger
        # 1. 推断文件系统的类型
        if fs_type is None:
            fs = file_system.new_local_fs()
        else:
            fs = file_system.instance_by_name(fs_type)

        # 2. 如果是url先把模型下载到本地
        if path.startswith("model://"):
            logger.info("从模型地址协议中加载模型。")
            # !! model_repo_client 与 DCPipeline形成双向依赖
            # 解决方法：DCPipeline仅支持从文件系统中加载，model_repo_client 处理 "model://" 协议

            # 2.1. 下载模型
            # 文件结构：
            #  /tmp/1/model.zip
            #  /tmp/1/model/{pipeline_files}

            random_dir = "/tmp/models/%s" % str(uuid.uuid4())
            LocalFileSystem().make_dirs(random_dir)  # 在本地创建临时目录
            model_tmp_path = '%s/model.zip' % random_dir
            logger.info("下载模型到:%s" % model_tmp_path)
            if download_timeout is None:
                download_timeout = (30, 60)
            model_repo_client.get(model_uri=path, model_repo_server_portal=model_repo_server_portal, destination_path=model_tmp_path, timeout=download_timeout)

            # 2.2. 解压模型

            extract_path = "%s/model" % random_dir
            LocalFileSystem().make_dirs(extract_path)  # 在本地创建临时目录
            logger.info("解压模型到:%s" % extract_path)
            zip_util.extract(model_tmp_path, extract_path)
            path = extract_path

        # 3. 加载 pipeline meta 文件
        pipeline_meta = PipelineMetaData.load_from_json_str(str_util.to_str(fs.read_bytes(path + '/meta.json')))

        # 4. 把图的顺序转换成列表, 执行的时候也要按生成的列表顺序
        steps_id_name_mapping = {}
        for s_meta in pipeline_meta.steps_meta:
            steps_id_name_mapping[s_meta['id']] = s_meta['dirName']

        # 加载所有的step, 并根据step的输入类型是否是pySparkDataFrame，对step进行分片，连续的相同类型分到一个片里。
        step_ids = DCPipeline.convert_graph_to_list(pipeline_meta.graph)
        step_dir_names = [steps_id_name_mapping[_id] for _id in step_ids]
        logger.info("要加载的模型step有：{}".format(str(step_dir_names)))

        # 2021/6/10 镜像升级sklearn版本从0.20.3到0.23.2，兼容老模型 ===start===
        # 先删除modules中的sklearn，再把老版本的sklearn路径加到sys.path的第一的位置
        from dc_model_repo.step.base import StepMetaData
        is_dt = False
        for step_name in step_dir_names:
            step_path = os.path.join(path, DCPipeline.FILE_STEPS, step_name)
            step_meta_path = DCStep.serialize_meta_path(step_path)
            step_meta = StepMetaData.load_from_json_str(fs.read_bytes(step_meta_path))
            if step_meta.framework == "DeepTables":
                is_dt = True
                break

        if pipeline_meta.version != '6.1.0':
            import sys
            ms = sys.modules
            sklearn_ms = []

            for m in ms:
                if m.startswith('sklearn'):
                    sklearn_ms.append(m)
            logger.info('即将删除的skearn modules：{}'.format(sklearn_ms))
            for m in sklearn_ms:
                ms.pop(m)

            logger.info('删除skearn modules，检查当前是否包含skearn modules')
            for m in ms:
                if m.startswith('sklearn'):
                    logger.info(ms[m])

            # 把老版本的sklearn加到sys.path中
            sys.path.insert(0, '/opt/sklearn0.20.3/')
        # 2021/6/10 镜像升级sklearn版本从0.20.3到0.23.2，兼容老模型 ===end===

        RANGE_TYPE_TRANSFORMER = "transformer"
        RANGE_TYPE_ESTIMATOR = "estimator"
        StepRange = collections.namedtuple(typename="StepRange", field_names=["input_type", "step_names", "range_type"])
        step_path_mapping = dict()
        origin_steps = collections.OrderedDict()

        # 初始化第一个range
        step_ranges = []  # [StepRange(input_type=pipeline_meta.input_type, step_names=step_dir_names[:1], range_type=RANGE_TYPE_TRANSFORMER)]

        for i, step_name in enumerate(step_dir_names):
            # 加载step
            try:
                # 先加载这个step的meta信息：
                step_path = os.path.join(path, DCPipeline.FILE_STEPS, step_name)
                step_meta_path = DCStep.serialize_meta_path(step_path)
                step_meta = json.loads(fs.read_bytes(step_meta_path))
                step_input_type = step_meta["inputType"]
                # 如果step的input_type是spark DataFrame， 这时不反序列化模型，只设置一个空壳对象。
                if step_input_type == DatasetType.PySparkDataFrame:
                    step = FakeSparkStep(input_type=step_input_type)
                # 如果不是spark的DataFrame，直接反序列化模型。
                else:
                    step = DCStep.load(fs_type=fs_type, path=step_path)
            except Exception as e:
                logger.error("加载DCStep失败，地址: %s" % step_path)
                raise e
            step_path_mapping[step] = step_path
            origin_steps[step_name] = step

            # 维护step分片
            # 当前step输入为spark时候
            if step.input_type == DatasetType.PySparkDataFrame:
                if len(step_ranges) > 0 and step_ranges[-1].input_type == DatasetType.PySparkDataFrame:
                    step_ranges[-1].step_names.append(step_name)
                else:
                    step_ranges.append(StepRange(input_type=step.input_type, step_names=[step_name], range_type=RANGE_TYPE_TRANSFORMER))
            # 当前step输入非spark时候
            else:
                if len(step_ranges) > 0 and step_ranges[-1].input_type != DatasetType.PySparkDataFrame:
                    step_ranges[-1].step_names.append(step_name)
                else:
                    step_ranges.append(StepRange(input_type=step.input_type, step_names=[step_name], range_type=RANGE_TYPE_TRANSFORMER))

        # 将最后一个range置为estimator类型，因为目前加载的pipeline最后一个步骤肯定是estimator。为后续扩展调用spark组成的transformer做准备
        step_ranges[-1] = step_ranges[-1]._replace(range_type=RANGE_TYPE_ESTIMATOR)

        pipeline_skip_validation_flag = step_ranges[0].input_type == DatasetType.PySparkDataFrame

        steps = []
        for step_range in step_ranges:
            if step_range.input_type == DatasetType.PySparkDataFrame:
                if step_range.input_type == DatasetType.PySparkDataFrame:
                    # 组装能调用mrsdk-lib接口的Step
                    logger.info("这些step将合并成Java模型：{}，类型：{}".format(step_range.step_names, step_range.range_type))
                    if step_range.range_type == RANGE_TYPE_ESTIMATOR:
                        cur_steps = collections.OrderedDict([(s, origin_steps[s]) for s in step_range.step_names])
                        java_model = GeneratedJavaEstimator(steps=cur_steps, model_path=path, jar_path=jar_path, debug_log=debug_log)
                        steps.append(java_model)
                    else:
                        # 后续支持spark类型的transformer时在这里添加逻辑
                        raise Exception("现在还不支持spark类型的transformer。")
            else:
                steps.extend([origin_steps[c] for c in step_range.step_names])

        # 6. 加载评估数据(仅加载工作流中生成的评估数据)
        performance_meta_path = path + "/performance/performance/performance.json"
        if fs.exists(performance_meta_path):
            performance_list = json_util.to_object(str_util.to_str(fs.read_bytes(performance_meta_path)))
            performance = [ChartData.load_from_dict(p) for p in performance_list]
        else:
            performance = None

        # 7. 创建Pipeline
        pipeline = DCPipeline(steps=steps,
                              name=pipeline_meta.name,
                              learning_type=pipeline_meta.learning_type,
                              input_type=pipeline_meta.input_type,
                              # labels=pipeline_meta.labels,
                              input_features=pipeline_meta.input_features,
                              default_metric=pipeline_meta.default_metric,
                              attachments=pipeline_meta.attachments,
                              performance=performance,
                              description=pipeline_meta.description,
                              skip_validation=pipeline_skip_validation_flag,
                              load_path=path,
                              pipeline_id=pipeline_meta.id)

        pipeline.step_path_mapping = step_path_mapping
        # 8. 设置样本数据路径
        # pipeline.target_sample_data_path = pipeline_meta.sample_data_path
        # pipeline.sample_data_path = pipeline_meta.target_sample_data_path

        return pipeline


class StepConnection(DictSerializable):

    def __init__(self, source_step_id, source_io_name, target_io_name, target_step_id):
        self.source_step_id = source_step_id
        self.source_io_name = source_io_name
        self.target_io_name = target_io_name
        self.target_step_id = target_step_id

    @classmethod
    def field_mapping(cls):
        return {'source_step_id': 'sourceStepId',
                'source_io_name': 'sourceIOName',
                'target_io_name': 'targetIOName',
                'target_step_id': 'targetStepId'}

    def to_dict(self):
        return self.member2dict(self.field_mapping())

    @classmethod
    def load_from_dict(cls, dict_data):
        sc = StepConnection(None, None, None, None)
        sc.dict2member(cls.field_mapping(), dict_data)
        return sc


class BuildPMMLResult(object):
    def __init__(self, pmml_bytes, include_steps):
        self.pmml_bytes = pmml_bytes
        self.include_steps = include_steps


class BuildSKLearnPipelineResult(object):
    def __init__(self, sklearn_pipeline, include_steps):
        self.sklearn_pipeline = sklearn_pipeline
        self.include_steps = include_steps
