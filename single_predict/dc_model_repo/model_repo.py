# -*- encoding: utf-8 -*-

"""
通过model_repo可以便捷地将sklearn spark等框架的算法封装成DCStep。
"""

from dc_model_repo.util import validate_util, cls_util
from dc_model_repo.sklearn_pandas.dataframe_mapper import DataFrameMapper as DCDataFrameMapper
from dc_model_repo.step.sklearn_step import SKLearnDCTransformer, SKLearnDCEstimator, SKLearnDCTuningEstimator, SKLearnDCCluster, SKLearnCommonUtil
from dc_model_repo.step.spark_step import SparkDCEstimator, SparkDCTransformer, SparkDCTuningEstimator

import pandas as pd
import numpy as np


def fit_transform(operator=None, model=None, X_train=None, y_train=None, input_cols=None, target_cols=None,
                  algorithm_name=None, options=None, extension=None, **kwargs):
    """SKLearn、PySpark、Keras、Tensorflow中的模型进行训练，并且对训练数据进行转换操作。

    Args:
        operator (object): 是SKLearn、PySpark、Keras、Tensorflow中的transformer或者estimator。
        X_train (DataFrame): 训练的特征数据, 可以是pandas或者pyspark的DataFrame.
        y_train (np.array): 训练的标签数据，当为spark模型或时可不写。
        input_cols (list): 指定训练时使用的特征列, 当为空时使用所有列进行训练。
        target_cols (list): 预测的目标列名，默认为prediction。
        algorithm_name (str): 算法名称，如果为空则使用模型的类名。
        options (dict): 模型训练参数。
        **kwargs: 扩展参数。

    Returns:
        step: DCStep 实例。
        df: 转换之后的数据。
    """
    step = fit(operator=operator, model=model, input_cols=input_cols, X_train=X_train, y_train=y_train, target_cols=target_cols,
               algorithm_name=algorithm_name, options=options, extension=extension, **kwargs)
    df = step.transform(X_train)
    return step, df


def __create_sklearn_transformer_step__(operator, output_type, input_cols, algorithm_name, extension, **kwargs):
    return SKLearnDCTransformer(operator=operator,
                                input_cols=input_cols,
                                algorithm_name=algorithm_name,
                                extension=extension,
                                output_type=output_type,
                                **kwargs)


def __create_sklearn_estimator_step__(operator, output_type, input_cols, output_cols, target_cols, algorithm_name, extension):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    if isinstance(operator, GridSearchCV) or isinstance(operator, RandomizedSearchCV):
        dc_cls = SKLearnDCTuningEstimator
    else:
        dc_cls = SKLearnDCEstimator

    estimator = dc_cls(operator=operator,
                       input_cols=input_cols,
                       output_col=output_cols[0],
                       target_col=target_cols[0],
                       algorithm_name=algorithm_name,
                       output_type=output_type,
                       extension=extension)
    return estimator


def fit(operator=None, model=None, X_train=None, y_train=None, input_cols=None, target_cols=None, output_cols=None, algorithm_name=None,
        options=None, output_type=None, extension=None, learning_type=None, positive_label=None, binary_threshold=None, **kwargs):
    """SKLearn、PySpark、Keras、Tensorflow中的模型进行训练，并转换为aps的DCStep。

    Args:

        operator (object):  SKLearn、PySpark、Keras、Tensorflow中的transformer或者estimator;
                         当为SKLearn的树或者线性模型时候会生成该模型的可视化结构，对于SKLearn 和PySpark也会尝试获取特征重要性数据。
        X_train (DataFrame): 训练的特征数据, 可以是pandas或者pyspark或者dask的DataFrame、或dict数据类型.
        y_train (np.array): 训练的标签数据，当为spark模型或时可不写。
        input_cols (list): 指定训练时使用的特征列, 当为空时使用所有列进行训练。
        target_cols (list): 训练的标签列名，对表格数据，当为estimator有效，默认为label。
        output_cols (list): 预测输出的列名，对表格数据，当为estimator有效，默认为prediction。
        algorithm_name (str): 算法名称，如果为空则使用模型的类名。
        options (dict): 模型训练参数，只有带内置的model的Step才会用到，送给内置model的的类的fit方法。
        output_type： 指定的输出类型。如果为None，后续程序执行时会设置成跟input_type一致的。
        extension: pass
        learning_type: 任务类型，只有Estimator时候才有用
        positive_label: 正样本标签，只有二分类Estimator才有用
        binary_threshold: 正样本阈值，大于等于该阈值时预测为正样本
        **kwargs: 扩展参数。

    Returns:
        step: DCStep 实例。

    References:
      https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    """
    from dc_model_repo.base.mr_log import logger

    # 1. 校验参数类型
    if model is not None:
        if operator is None:
            logger.warning("参数\"model\"将被废弃，请使用operator代替。")
            operator = model

    if X_train is None:
        raise Exception("参数\"X_train\"不能为空。")

    # 2. 设置默认参数
    if target_cols is None or len(target_cols) == 0:
        target_cols = ["label"]

    if output_cols is None or len(output_cols) == 0:
        output_cols = ['prediction']

    if input_cols is None:
        if isinstance(X_train, pd.DataFrame):
            input_cols = X_train.columns.tolist()
            if validate_util.is_non_empty_list(input_cols):
                # In case of col name is not str, should convert to str before invoke ",".join
                logger.info("未指定训练列，将使用所有列训练, 检测到训练数据有以下列: %s" % ",".join([str(x) for x in input_cols]))
            else:
                raise Exception("训练数据不能为空。")
        elif isinstance(X_train, dict):
            input_cols = [c for c in X_train]
            if len(input_cols) == 0:
                raise Exception("训练数据不能为空")

    kwargs["learning_type"] = learning_type
    kwargs["positive_label"] = positive_label
    kwargs["binary_threshold"] = binary_threshold

    # 3. 解析出对象的包名、类名
    module_name = cls_util.get_module_name(operator)
    class_name = cls_util.get_class_name(operator)
    full_class_name = cls_util.get_full_class_name(operator)
    package_array = module_name.split(".")
    len_pkg_array = len(package_array)
    first_pkg_name = package_array[0]

    # 4. 判断是否为DT模型(有的模块可能没装dt, 所以判断类名)
    if full_class_name == "deeptables.models.deeptable.DeepTable":
        from dc_model_repo.step.deeptables_step import DeepTablesDCStep  # 此处导入，防止初始化类时加载deeptables失败（在没有dt时）
        step = DeepTablesDCStep(operator=operator,
                                input_cols=input_cols,
                                target_col=target_cols[0],
                                output_col=output_cols[0],
                                extension=extension,
                                output_type=output_type)
        return step.fit(X_train, y_train, options=options)

    # 5. 支持内置的DataFrameMapper
    if isinstance(operator, DCDataFrameMapper):
        sp = __create_sklearn_transformer_step__(operator, output_type, input_cols, algorithm_name, extension, **kwargs)

        return sp.fit(X_train, y_train, options=options, **kwargs)

    # 6. 支持SKLearn 框架
    if first_pkg_name == 'sklearn':
        if len_pkg_array > 1:
            # if not isinstance(X_train, (pd.DataFrame, dict)):
            #     raise Exception("使用sklearn框架时候X_train只能为pandas DatafFrame类型或Dict类型，现在却为[{}]".format((type(X_train))))
            part_two = package_array[1]
            validate_util.require_list_non_empty('input_cols', input_cols)

            if part_two in ['preprocessing', 'impute', 'feature_selection']:
                if part_two == "feature_selection":
                    sp = __create_sklearn_transformer_step__(operator, output_type, input_cols, algorithm_name, extension, preserve_origin_dtypes=True, **kwargs)
                else:
                    sp = __create_sklearn_transformer_step__(operator, output_type, input_cols, algorithm_name, extension, **kwargs)
            elif part_two in ["ensemble"]:
                # 融合模型
                if hasattr(operator, 'predict'):
                    sp = __create_sklearn_estimator_step__(operator, output_type, input_cols, output_cols, target_cols, algorithm_name, extension)
                else:
                    try:
                        from sklearn.ensemble import StackingClassifier, StackingRegressor
                    except Exception as e:
                        raise Exception("无法验证模型是否为StackingClassifier，请检查当前SKLearn 版本是否高于v0.22：%s" % str(operator))

                    # StackingClassifier fit之后才有predict方法。
                    if isinstance(operator, StackingClassifier) or isinstance(operator, StackingRegressor):
                        sp = __create_sklearn_estimator_step__(operator, output_type, input_cols, output_cols, target_cols, algorithm_name, extension)
                    else:
                        raise Exception("该融合模型不是StackingClassifier，且没有predict方法: %s" % str(operator))
            elif SKLearnCommonUtil.is_supported_clustering(operator):
                sp = SKLearnDCCluster(operator=operator, input_cols=input_cols, output_col=output_cols[0], target_col=target_cols[0], algorithm_name=algorithm_name, extension=extension)
            elif hasattr(operator, 'predict'):
                sp = __create_sklearn_estimator_step__(operator, output_type, input_cols, output_cols, target_cols, algorithm_name, extension)
            else:
                raise Exception("无法识别的SKLearn模型: %s" % str(operator))
            return sp.fit(X_train, y_train, options=options, **kwargs)
        else:
            raise Exception("无法识别模块名： %s" % module_name)

    # 7. 支持外部的DataFrameMapper
    if first_pkg_name == 'sklearn_pandas':
        if len_pkg_array > 1:
            if package_array[1] == 'dataframe_mapper':
                validate_util.require_list_non_empty('input_cols', input_cols)
                spt = __create_sklearn_transformer_step__(operator, output_type, input_cols, algorithm_name, extension, **kwargs)
                return spt.fit(X_train, y_train, options=options)

    # 8. 支持 xgboost
    if first_pkg_name == 'xgboost':
        spe = __create_sklearn_estimator_step__(operator, output_type, input_cols, output_cols, target_cols, algorithm_name, extension)
        return spe.fit(X_train, y_train, options=options, **kwargs)

    # 9. 支持 pyspark
    if first_pkg_name == 'pyspark':
        # 判断是否为调优
        from pyspark.ml.tuning import CrossValidator, TrainValidationSplit
        if isinstance(operator, CrossValidator) or isinstance(operator, TrainValidationSplit):
            operator = SparkDCTuningEstimator(operator, algorithm_name=algorithm_name, output_type=output_type)
        # 提取input_cols
        elif len_pkg_array > 2:
            if package_array[1] == "ml" and package_array[2] == "feature":
                operator = SparkDCTransformer(operator=operator, algorithm_name=algorithm_name, output_type=output_type)
            else:
                operator = SparkDCEstimator(operator=operator, algorithm_name=algorithm_name, output_type=output_type)
        else:
            operator = SparkDCEstimator(operator=operator, algorithm_name=algorithm_name, output_type=output_type)
        return operator.fit(X_train, y_train, options=options, **kwargs)

    # 10. 支持keras
    if first_pkg_name.find('keras') != -1:
        from dc_model_repo.step.keras_step import KerasDCCustomerEstimator
        operator = KerasDCCustomerEstimator(model=operator, input_cols=input_cols, target_cols=target_cols,
                                            extension=extension)
        operator.output_type = output_type
        return operator.fit(X_train, y_train, **kwargs)

    # 11. 支持 PipelineTransformerStep
    if len_pkg_array > 2 and package_array[2] == "pipeline_transformer_step":
        from dc_model_repo.step.pipeline_transformer_step import PipelineTransformerStep
        operator = PipelineTransformerStep(operator=operator, algorithm_name=algorithm_name, output_type=output_type)
        return operator.fit(X_train)

    # 12. 支持贝叶斯优化包
    if first_pkg_name == "skopt":
        from dc_model_repo.step.skopt_step import SkoptDCTuningEstimator
        operator = SkoptDCTuningEstimator(operator=operator,
                                          input_cols=input_cols,
                                          output_col=output_cols[0],
                                          target_col=target_cols[0],
                                          algorithm_name=algorithm_name,
                                          output_type=output_type,
                                          extension=extension)
        return operator.fit(X_train, y_train, options=options, **kwargs)

    raise Exception("SDK还不支持处理类\"%s\"，请关注后续版本升级。" % cls_util.get_full_class_name(operator))
