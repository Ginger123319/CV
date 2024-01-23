# -*- encoding: utf-8 -*-

from dc_model_repo.step.base import BaseTransformer, BaseEstimator
import os
import time
from dc_model_repo.base import StepType, FrameworkType, Field, Output, ChartData, ModelFileFormatType, LearningType
from dc_model_repo.util import cls_util, validate_util, operator_output_util
import numpy as np
from dc_model_repo.base import Param
import pandas as pd


class SKLearnCommonUtil:

    @staticmethod
    def is_supported_clustering(operator):
        from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, Birch
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        return isinstance(operator, (KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, Birch, GaussianMixture, BayesianGaussianMixture))

    @staticmethod
    def silhouette_score(estimator, cur_x):
        from sklearn.metrics import silhouette_score
        # 为了持久化这个方法，需要定义在顶层first level，不能定义在方法里
        y_pred = estimator.predict(cur_x)
        import numpy as np
        if len(np.unique(y_pred)) < 2:
            # 当簇点<2时候没法使用silhouette算法
            return -1
        else:
            return silhouette_score(cur_x, y_pred)


class SKLearnDCTransformer(BaseTransformer):
    """SKLearn中所有Transformer的基类。把SKLearn的transformer封装成Step。

    Args:
        operator (object): SKLearn的transformer或者estimator。
        dfm_model: 如果需要包装成DataFrameMapper, 表示DFM。
        model (object): APS 封装后支持对列处理的模型。
        kind (str): Step的类型，见 `dc_model_repo.base.StepType`
        input_cols (list): Step处理的列。
        algorithm_name (str): 算法名称。
        extension (dict): 扩展信息字段。
        use_df_wrapper: 是否使用dataframe_wrapper对模型进行包装，只有operator为sklearn原始模型时有效。默认：True，进行包装。
        **kwargs: 备用
    """

    def __init__(self, operator, input_cols, algorithm_name=None, extension=None, use_df_wrapper=True, **kwargs):

        # 1. 使用dfm包装来支持列名
        self.dfm_model = None
        from sklearn_pandas.dataframe_mapper import DataFrameMapper as SKLearnDataFrameMapper
        from dc_model_repo.sklearn_pandas.dataframe_mapper import DataFrameMapper as DCDataFrameMapper
        if isinstance(operator, SKLearnDataFrameMapper):
            from dc_model_repo.base.mr_log import logger
            logger.warning("当前使用dfm进行训练，请确保已经设置参数input_df=True, default=None, df_out=True，建议使用" +
                           "dc_model_repo.sklearn_pandas.DataFrameMapper替代。")
        elif isinstance(operator, DCDataFrameMapper):
            self.dfm_model = operator
        else:
            # 包装成dfm
            if type(operator).__module__.startswith("sklearn") and not use_df_wrapper:
                self.dfm_model = None
            else:
                preserve_origin_dtypes = kwargs.pop("preserve_origin_dtypes", False)
                self.dfm_model = DCDataFrameMapper(features=[(input_cols, operator)], input_df=True, default=None, df_out=True, preserve_origin_dtypes=preserve_origin_dtypes)

        # 2. 调用父类构造方法
        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)

        super(SKLearnDCTransformer, self).__init__(operator=operator,
                                                   framework=FrameworkType.SKLearn,
                                                   model_format=ModelFileFormatType.PKL,
                                                   input_cols=input_cols,
                                                   algorithm_name=algorithm_name,
                                                   extension=extension,
                                                   **kwargs)
        self.model_path = 'data/model.pkl'

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator"]

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pkl')
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize2bytes(self.model)
        fs.write_bytes(model_path, obj_bytes)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/data/model.pkl" % step_path
        logger.info("开始加载SKLearn模型在: %s." % model_path)
        t1 = time.time()
        from dc_model_repo.util import pkl_util

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(file_system.FS_LOCAL)

        self.model = pkl_util.deserialize(fs.read_bytes(model_path))
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def transform_data(self, X, **kwargs):
        return self.model.transform(X)

    def get_params(self):
        """解析SKLearn模型的参数， 使用训练前的原始模型。
        Returns:
        """
        return self.get_params_from_dict_items(self.operator.__dict__, self.input_cols)

    def fit_model(self, X, y=None, options=None, **kwargs):
        if self.dfm_model is not None:
            return self.fit_input_model(self.dfm_model, X, y, options)
        else:
            return super(SKLearnDCTransformer, self).fit_model(X, y, options, **kwargs)


class SKLearnLikePredictDCEstimator(BaseEstimator):
    """
    This class is preserved only for compatibility of APS31CustomStep.It will be removed later.

    把SKLearn的transformer封装成Step。

    Args:
        operator (object): SKLearn的transformer或者estimator。
        dfm_model: 如果需要包装成DataFrameMapper, 表示DFM。
        model (object): APS 封装后支持对列处理的模型。
        kind (str): Step的类型，见 `dc_model_repo.base.StepType`
        input_cols (list): Step处理的列。
        algorithm_name (str): 算法名称。
        extension (dict): 扩展信息字段。
        **kwargs:
    """

    def __init__(self, operator, input_cols, algorithm_name=None, framework=None, model_format=None, extension=None, **kwargs):

        import warnings
        warnings.warn("This class [SKLearnLikePredictDCEstimator] is preserved only for compatibility of [APS31CustomStep]. It will be removed later.", category=DeprecationWarning)

        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)
        if framework is None:
            framework = FrameworkType.SKLearn
        if model_format is None:
            model_format = ModelFileFormatType.PKL

        super(SKLearnLikePredictDCEstimator, self).__init__(operator=operator,
                                                            framework=framework,
                                                            model_format=model_format,
                                                            input_cols=input_cols,
                                                            algorithm_name=algorithm_name,
                                                            extension=extension,
                                                            **kwargs)
        self.model_path = 'data/model.pkl'

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator"]

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pkl')
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize2bytes(self.model)
        fs.write_bytes(model_path, obj_bytes)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/data/model.pkl" % step_path
        logger.info("开始加载SKLearn模型在: %s." % model_path)
        t1 = time.time()
        from dc_model_repo.util import pkl_util

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(file_system.FS_LOCAL)

        self.model = pkl_util.deserialize(fs.read_bytes(model_path))
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def transform_data(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        """转换数据。

        Args:
            X:
            calc_max_proba: 是否计算最大概率，如果模型输出概率，则计算。
            calc_all_proba: 是否计算每一个label的概率，如果模型输出概率，则计算。
            **kwargs:

        Returns:

        """
        from dc_model_repo.base.mr_log import logger

        # 1. xgboost 预测时需要和inputfeatures中列顺序一致
        if self.algorithm_name.startswith("XGB"):
            names = [input.name for input in self.input_features]
            X = X[names]

        # 2. 检查参数
        if calc_all_proba is True and calc_max_proba is False:
            logger.warning("已经设置calc_all_proba为True, 忽略calc_max_proba为False。")
            calc_max_proba = True

        output = self.outputs[0]

        # 3. 预测
        prediction = self.model.predict(X)

        # 4. 计算概率
        proba = None
        if calc_max_proba:  # 只要满足任何一个条件就计算
            if hasattr(self.model, 'predict_proba'):
                # 4.1. 有predict_proba但是不一定调用成功。voting 算法，22.x版本以前为hard时 此方法不能用
                try:
                    proba = self.model.predict_proba(X)
                except Exception as e:
                    logger.error("调用predict_proba方法失败，跳过概率计算。")
                    logger.error(e)
                    proba = None
            else:
                logger.warning("设置计算概率，但是模型没有predict_proba方法。")

        X = operator_output_util.make_predict_output_data_frame(X, prediction, proba, calc_all_proba, output)

        return X

    def predict(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        return self.transform(X, calc_max_proba=calc_max_proba, calc_all_proba=calc_all_proba, remove_unnecessary_cols=True, **kwargs)

    def get_params(self):
        """解析SKLearn模型的参数， 使用训练前的原始模型。

        Returns:
            list: 返回模型的参数, 数组内元素的类型为 :class:`dc_model_repo.base.Param`
        """
        return self.get_params_from_dict_items(self.operator.__dict__, self.input_cols)


class SKLearnDCEstimator(BaseEstimator):
    """把SKLearn的estimator封装成DC的estimator

    Args:
        operator (object): SKLearn的estimator。
        dfm_model: 如果需要包装成DataFrameMapper, 表示DFM。
        model (object): APS 封装后支持对列处理的模型。
        kind (str): Step的类型，见 `dc_model_repo.base.StepType`
        input_cols (list): Step处理的列。
        algorithm_name (str): 算法名称。
        extension (dict): 扩展信息字段。
        **kwargs:
    """

    def __init__(self, operator, input_cols, target_col, output_col, algorithm_name=None, extension=None, **kwargs):

        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)
        # 1. 调用父类构造方法
        super(SKLearnDCEstimator, self).__init__(operator=operator,
                                                 input_cols=input_cols,
                                                 algorithm_name=algorithm_name,
                                                 target_cols=[target_col],
                                                 output_cols=[output_col],
                                                 framework=FrameworkType.SKLearn,
                                                 model_format=ModelFileFormatType.PKL,
                                                 extension=extension,
                                                 **kwargs)

        # 2. 初始化变量
        self.labels = None  # labels 的值，不是label_col
        self.model_path = 'data/model.pkl'

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator"]

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pkl')
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize2bytes(self.model)
        fs.write_bytes(model_path, obj_bytes)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/data/model.pkl" % step_path
        logger.info("开始加载SKLearn模型在: %s." % model_path)
        t1 = time.time()
        from dc_model_repo.util import pkl_util

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(file_system.FS_LOCAL)

        self.model = pkl_util.deserialize(fs.read_bytes(model_path))
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def transform_data(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        """转换数据。

        Args:
            X:
            calc_max_proba: 是否计算最大概率，如果模型输出概率，则计算。
            calc_all_proba: 是否计算每一个label的概率，如果模型输出概率，则计算。
            **kwargs: 备用

        Returns:

        """
        from dc_model_repo.base.mr_log import logger

        # 1. xgboost 预测时需要和inputfeatures中列顺序一致
        if self.algorithm_name.startswith("XGB"):
            names = [input.name for input in self.input_features]
            X = X[names]

        # 2. 检查参数
        if calc_all_proba is True and calc_max_proba is False:
            logger.warning("已经设置calc_all_proba为True, 忽略calc_max_proba为False。")
            calc_max_proba = True

        output = self.outputs[0]

        # 3. 预测
        prediction = self.model.predict(X)

        # 4. 计算概率
        proba = None
        if calc_max_proba:  # 只要满足任何一个条件就计算
            if hasattr(self.model, 'predict_proba'):
                # 4.1. 有predict_proba但是不一定调用成功。voting 算法，22.x版本以前为hard时 此方法不能用
                try:
                    proba = self.model.predict_proba(X)
                except Exception as e:
                    logger.error("调用predict_proba方法失败，跳过概率计算。")
                    logger.error(e)
                    proba = None
            else:
                logger.warning("设置计算概率，但是模型没有predict_proba方法。")

        X = operator_output_util.make_predict_output_data_frame(X, prediction, proba, calc_all_proba, output, preserve_origin_cols=kwargs.get("preserve_origin_cols", False))

        return X

    def predict(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        return self.transform(X, calc_max_proba=calc_max_proba, calc_all_proba=calc_all_proba, remove_unnecessary_cols=True, **kwargs)

    def get_params(self):
        """解析SKLearn模型的参数， 使用训练前的原始模型。

        Returns:
            list: 返回模型的参数, 数组内元素的类型为 :class:`dc_model_repo.base.Param`
        """
        return self.get_params_from_dict_items(self.operator.__dict__, self.input_cols)

    def persist_explanation(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        from dc_model_repo.util import validate_util

        explanation = []
        model = self.model

        def get_classes_from_model():
            if hasattr(model, 'classes_'):
                return model.classes_
            else:
                return None

        # 1. 对树结构特殊处理落地成文件
        from dc_model_repo.step import tree_visual
        if tree_visual.is_tree_model(self.model):
            classes_ = get_classes_from_model()
            validate_util.require_list_non_empty('train_columns', self.train_columns)
            try:
                visual_tree_data_list = tree_visual.build_trees_visual_data(self.model, self.serialize_explanation_path(
                    destination), self.train_columns, classes_)
                explanation.append(visual_tree_data_list)
            except Exception as e:
                logger.exception(e)
                logger.error("生成模型可视化失败。")
        else:
            logger.info("当前模型不是树模型,不能解析相关可视化参数。")

        # 2. 创建模型可解释
        from dc_model_repo.step import regression_coefficients
        # 2.1. 检测是否为线性回归
        if regression_coefficients.is_linear_model(self.model):
            classes_ = get_classes_from_model()
            validate_util.require_list_non_empty('train_columns', self.train_columns)
            try:
                rc_visual_data = regression_coefficients.build_visual_data(self.model, self.train_columns, class_names=classes_)
                cd = ChartData('regressionCoefficients', 'regressionCoefficients', rc_visual_data)
                explanation.append(cd)
            except Exception as e:
                logger.exception(e)
                logger.error("生成模型可视化失败。")
        else:
            logger.info("当前模型不是线性模型,不能解析相关可视化参数。")

        # 3. 解析模型的特征重要性
        if hasattr(self.model, 'feature_importances_'):
            # 3.1. 读取数据列和特征重要性
            feature_importances_ = self.model.feature_importances_
            if validate_util.is_non_empty_list(self.train_columns) and validate_util.is_non_empty_list(feature_importances_):
                if len(self.train_columns) == len(feature_importances_):
                    cd = self.get_feature_importances(self.train_columns, feature_importances_, 'float64')
                    explanation.append(cd)
                else:
                    logger.warning("训练数据列和特征重要性长度不相同。")
            else:
                logger.info("训练数据列、或特征重要性为空。")
        else:
            logger.info("当前模型中没有特征重要性的数据。")

        # 4. 写入数据
        self.persist_explanation_object(fs, destination, explanation)

    def cast_as_df(self, data):
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame(data=data)
        else:
            return data

    def get_as_pd_data_type(self, data):
        """将输出的numpy数据转换成dataframe，然后获取其类型。

        Args:
            data: numpy数组，并且只有一列。

        Returns:
            str: 类型名称
        """
        df = self.cast_as_df(data)
        return list(df.dtypes.to_dict().values())[0].name

    def get_targets(self, x, y=None, options=None, **kwargs):
        target_name = self.target_cols[0]
        output_field_type = self.get_as_pd_data_type(y)
        return [Field(target_name, output_field_type)]

    def get_outputs(self, x, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        # 1. 解析输出列的名称和类型
        output_data_type = self.get_as_pd_data_type(y)
        output_name = self.output_cols[0]  # 用户设置的输出列名称

        if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'classes_'):
            self.labels = self.model.classes_
        else:
            logger.info("模型=%s没有predict_proba方法或者不是分类模型，不生成概率列。" % str(self.model))

        output = operator_output_util.make_output(output_name, output_data_type, self.labels, 'float64')

        return [output]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        df_y = self.cast_as_df(y)
        return self.get_data_sampler(df_y)

    def fit_post(self, X, y=None, options=None, **kwargs):
        self.train_columns = X.columns


class SKLearnDCCluster(BaseEstimator):
    """Deal with sklearn's cluster method which can be used to persisted and predict for new samples.

    Args:
        operator: 传入的算子，会对这个算子执行.fit(X, y, **options)
        input_cols: 当前算子要处理的列，在fit过程会取X的列与其交集当设置到self.input_features
        output_col: 输出列，list类型，如果为None或[]，会设置成默认值["prediction"]。
          这个属性会通过get_outputs转为self.outputs，默认的get_outputs只支持一个元素，如果需要输出多列，需要复写get_outputs方法。
        algorithm_name: 算法名称
        extension: 扩展信息字段
        **kwargs: 预留参数位置
    """

    def __init__(self, operator, input_cols, output_col, target_col=None, algorithm_name=None, extension=None, **kwargs):

        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator)
        # 1. 调用父类构造方法
        from dc_model_repo.base import LearningType
        super(SKLearnDCCluster, self).__init__(operator=operator,
                                               input_cols=input_cols,
                                               algorithm_name=algorithm_name,
                                               target_cols=[target_col],
                                               output_cols=[output_col],
                                               framework=FrameworkType.SKLearn,
                                               model_format=ModelFileFormatType.PKL,
                                               extension=extension,
                                               learning_type=LearningType.Clustering,
                                               **kwargs)

        # 2. 初始化变量
        self.labels = None  # labels 的值，不是label_col
        self.model_path = 'data/model.pkl'

    def get_persist_step_ignore_variables(self):
        return ["model"]

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pkl')
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize2bytes(self.model)
        fs.write_bytes(model_path, obj_bytes)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/data/model.pkl" % step_path
        logger.info("开始加载SKLearn模型在: %s." % model_path)
        t1 = time.time()
        from dc_model_repo.util import pkl_util

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(file_system.FS_LOCAL)

        self.model = pkl_util.deserialize(fs.read_bytes(model_path))
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def predict(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        # 1. 校验输入的类型是否匹配并去除无关数据
        from dc_model_repo.util import dataset_util
        X = dataset_util.validate_and_cast_input_data(X, self.input_type, self.input_features, remove_unnecessary_cols=True)
        # 2. 预测
        prediction = self.model.predict(X)
        # 3. Generate result
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X)
        X = operator_output_util.make_predict_output_data_frame(input_df=X, prediction=prediction, proba=None, calc_all_proba=False, output=self.outputs[0], preserve_origin_cols=kwargs.get("preserve_origin_cols", False))
        return X

    def get_params(self):
        """解析SKLearn模型的参数， 使用训练前的原始模型。

        Returns:
            list: 返回模型的参数, 数组内元素的类型为 :class:`dc_model_repo.base.Param`
        """
        return self.get_params_from_dict_items(self.operator.__dict__, self.input_cols)

    def persist_explanation(self, fs, destination):
        return None

    def get_targets(self, x, y=None, options=None, **kwargs):
        return None

    def get_outputs(self, x, y=None, options=None, **kwargs):
        output = Output(name=self.output_cols[0], type="unknown")
        return [output]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        df_y = y if isinstance(y, pd.DataFrame) else pd.DataFrame(data=y)
        return self.get_data_sampler(df_y)


class SKLearnDCTuningEstimator(BaseEstimator):
    """把SKLearn的用于调参的estimator封装成DC的TuningEstimator

    Args:
        operator (object): SKLearn的estimator。
        dfm_model: 如果需要包装成DataFrameMapper, 表示DFM。
        model (object): APS 封装后支持对列处理的模型。
        kind (str): Step的类型，见 `dc_model_repo.base.StepType`
        input_cols (list): Step处理的列。
        algorithm_name (str): 算法名称。
        extension (dict): 扩展信息字段。
    """

    def __init__(self, operator, input_cols, output_col, target_col, algorithm_name=None, extension=None, **kwargs):

        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        assert isinstance(operator, (GridSearchCV, RandomizedSearchCV))
        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator.estimator)

        self.tuning_estimator = operator  # 记录cv

        learning_type = kwargs.get("learning_type", None)
        if learning_type is None and SKLearnCommonUtil.is_supported_clustering(operator.estimator):
            learning_type = LearningType.Clustering

        # 1. 调用父类构造方法
        super(SKLearnDCTuningEstimator, self).__init__(operator=operator,
                                                       input_cols=input_cols,
                                                       algorithm_name=algorithm_name,
                                                       target_cols=[target_col],
                                                       output_cols=[output_col],
                                                       framework=FrameworkType.SKLearn,
                                                       model_format=ModelFileFormatType.PKL,
                                                       learning_type=learning_type,
                                                       extension=extension)

        # 2. 初始化变量
        self.labels = None  # labels 的值，不是label_col
        self.model_path = 'data/model.pkl'

    def get_persist_step_ignore_variables(self):
        return ["model", "operator", "tuning_estimator"]

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.pkl')
        from dc_model_repo.util import pkl_util
        obj_bytes = pkl_util.serialize2bytes(self.model)
        fs.write_bytes(model_path, obj_bytes)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/data/model.pkl" % step_path
        logger.info("开始加载SKLearn模型在: %s." % model_path)
        t1 = time.time()
        from dc_model_repo.util import pkl_util

        from dc_model_repo.base import file_system
        fs = file_system.instance_by_name(file_system.FS_LOCAL)

        self.model = pkl_util.deserialize(fs.read_bytes(model_path))
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def transform_data(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        """转换数据。

        Args:
            X:
            calc_max_proba: 是否计算最大概率，如果模型输出概率，则计算。
            calc_all_proba: 是否计算每一个label的概率，如果模型输出概率，则计算。
            **kwargs: 备用

        Returns:

        """
        from dc_model_repo.base.mr_log import logger

        # 1. xgboost 预测时需要和inputfeatures中列顺序一致  TODO k 想法把xgboost的实现抽出去
        if self.algorithm_name.startswith("XGB"):
            names = [input.name for input in self.input_features]
            X = X[names]

        # 2. 检查参数
        if calc_all_proba is True and calc_max_proba is False:
            logger.warning("已经设置calc_all_proba为True, 忽略calc_max_proba为False。")
            calc_max_proba = True

        output = self.outputs[0]

        # 3. 预测
        prediction = self.model.predict(X)

        # 4. 计算概率  TODO k 这里需要加上判断模型是分类还是回归的，如果是回归的，就不用执行概率计算环节了！
        proba = None
        if calc_max_proba:  # 只要满足任何一个条件就计算
            if hasattr(self.model, 'predict_proba'):
                # 4.1. 有predict_proba但是不一定调用成功。voting 算法，22.x版本以前为hard时 此方法不能用
                try:
                    proba = self.model.predict_proba(X)
                except Exception as e:
                    logger.error("调用predict_proba方法失败，跳过概率计算。")
                    logger.error(e)
                    proba = None
            else:
                logger.warning("设置计算概率，但是模型没有predict_proba方法。")

        X = operator_output_util.make_predict_output_data_frame(X, prediction, proba, calc_all_proba, output, preserve_origin_cols=kwargs.get("preserve_origin_cols", False))

        return X

    def predict(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        return self.transform(X, calc_max_proba=calc_max_proba, calc_all_proba=calc_all_proba, remove_unnecessary_cols=True, **kwargs)

    def persist_explanation(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        from dc_model_repo.util import validate_util

        explanation = []
        model = self.model

        def get_classes_from_model():
            if hasattr(model, 'classes_'):
                return model.classes_
            else:
                return None

        # 1. 对树结构特殊处理落地成文件
        from dc_model_repo.step import tree_visual
        if tree_visual.is_tree_model(self.model):
            classes_ = get_classes_from_model()
            validate_util.require_list_non_empty('train_columns', self.train_columns)
            try:
                visual_tree_data_list = tree_visual.build_trees_visual_data(self.model, self.serialize_explanation_path(
                    destination), self.train_columns, classes_)
                explanation.append(visual_tree_data_list)
            except Exception as e:
                logger.exception(e)
                logger.error("生成模型可视化失败。")
        else:
            logger.info("当前模型不是树模型,不能解析相关可视化参数。")

        # 2. 创建模型可解释
        from dc_model_repo.step import regression_coefficients
        # 2.1. 检测是否为线性回归
        if regression_coefficients.is_linear_model(self.model):
            classes_ = get_classes_from_model()
            validate_util.require_list_non_empty('train_columns', self.train_columns)
            try:
                rc_visual_data = regression_coefficients.build_visual_data(self.model, self.train_columns, class_names=classes_)
                cd = ChartData('regressionCoefficients', 'regressionCoefficients', rc_visual_data)
                explanation.append(cd)
            except Exception as e:
                logger.exception(e)
                logger.error("生成模型可视化失败。")
        else:
            logger.info("当前模型不是线性模型,不能解析相关可视化参数。")

        # 3. 解析模型的特征重要性
        if hasattr(self.model, 'feature_importances_'):
            # 3.1. 读取数据列和特征重要性
            feature_importances_ = self.model.feature_importances_
            if validate_util.is_non_empty_list(self.train_columns) and validate_util.is_non_empty_list(feature_importances_):
                if len(self.train_columns) == len(feature_importances_):
                    cd = self.get_feature_importances(self.train_columns, feature_importances_, 'float64')
                    explanation.append(cd)
                else:
                    logger.warning("训练数据列和特征重要性长度不相同。")
            else:
                logger.info("训练数据列、或特征重要性为空。")
        else:
            logger.info("当前模型中没有特征重要性的数据。")

        # 4. 写入数据
        self.persist_explanation_object(fs, destination, explanation)

    def cast_as_df(self, data):
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame(data=data)
        else:
            return data

    def get_as_pd_data_type(self, data):
        """将输出的numpy数据转换成dataframe，然后获取其类型。
        Args:
            data: numpy数组，并且只有一列。
        Returns:
        """
        df = self.cast_as_df(data)
        return list(df.dtypes.to_dict().values())[0].name

    def get_targets(self, x, y=None, options=None, **kwargs):
        if y is None:
            return None
        target_name = self.target_cols[0]
        output_field_type = self.get_as_pd_data_type(y)
        return [Field(target_name, output_field_type)]

    def get_outputs(self, x, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        # 1. 解析输出列的名称和类型

        if y is None and self.learning_type == LearningType.Clustering:
            return [Output(name=self.output_cols[0], type="unknown")]

        output_data_type = self.get_as_pd_data_type(y)
        output_name = self.output_cols[0]  # 用户设置的输出列名称

        if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'classes_'):
            self.labels = self.model.classes_
        else:
            logger.info("模型=%s没有predict_proba方法或者不是分类模型，不生成概率列。" % str(self.model))

        output = operator_output_util.make_output(output_name, output_data_type, self.labels, 'float64')

        return [output]

    def get_target_sample_data(self, x, y=None, options=None, **kwargs):
        df_y = self.cast_as_df(y)
        return self.get_data_sampler(df_y)

    def fit_post(self, X, y=None, options=None, **kwargs):
        self.train_columns = X.columns

    def get_params(self):
        """解析SKLearn模型的参数，从训练后的cv中解析出最优参数。

        Returns:
            list: 类型为 :class:`dc_model_repo.base.Param`
        """
        best_params = self.tuning_estimator.best_params_
        if best_params is not None:
            return [Param(k, None, best_params[k]) for k in best_params]
        else:
            return None

    def fit_model(self, X, y=None, options=None, **kwargs):
        # 训练模型
        self.tuning_estimator = self.fit_input_model(self.tuning_estimator, X, y, options, **kwargs)
        return self.tuning_estimator.best_estimator_
