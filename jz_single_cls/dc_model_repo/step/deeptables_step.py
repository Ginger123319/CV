# -*- encoding: utf-8 -*-

import os
import time
from os import path as P

import pandas as pd
from deeptables.models import deeptable

from dc_model_repo.base import Field
from dc_model_repo.base import FrameworkType, ModelFileFormatType, ChartData, Param
from dc_model_repo.step.base import BaseEstimator
from dc_model_repo.util import operator_output_util


class DeepTablesDCEstimator(BaseEstimator):

    def __init__(self, operator, input_cols, target_col, output_col, extension=None, **kwargs):
        """支持DeepTables 神经网络框架。
        注意该模块不支持与分布式算法同时运行，因为DeepTables不能从hdfs上加载模型。

        Args:
            operator (deeptables.models.deeptable.DeepTable):
            model (deeptables.models.deepmodel.DeepModel): DeepTable训练后的模型。
            input_cols (list): Step处理的列。
            target_col(str): 标签列。
            output_col(str): 预测的输出列。
            extension (dict): 扩展信息字段。
            **kwargs:
        """
        # 1. 调用父类构造方法
        super(DeepTablesDCEstimator, self).__init__(operator=operator,
                                                    framework=FrameworkType.DeepTables,
                                                    model_format=ModelFileFormatType.DIR,
                                                    input_cols=input_cols,
                                                    algorithm_name='DT神经网络',
                                                    requirements=["deeptables>=0.1.9"],
                                                    extension=extension,
                                                    target_cols=[target_col],
                                                    output_cols=[output_col],
                                                    **kwargs)

        # 2. 变量初始化
        self.model_path = 'data/model'  # 定义模型序列化后的位置

    def persist_model(self, fs, destination):
        model_path = P.join(self.serialize_data_path(destination), 'model')
        # 确保目录存在，dt创建会报错
        fs.make_dirs(model_path)
        self.model.save(model_path)

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        model_path = "%s/%s" % (step_path, self.model_path)
        logger.info("开始加载DeepModel模型在: %s." % model_path)
        t1 = time.time()
        self.model = deeptable.DeepTable.load(model_path)  # 只支持从本地加载
        self.operator = self.model
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def transform_data(self, X, calc_max_proba=True, calc_all_proba=False, **kwargs):
        """转换数据。

        Args:
            X:
            calc_max_proba: 是否计算最大概率。对于分类任务时候有效，其他任务设置了也不会有概率列。
            calc_all_proba:
            **kwargs:

        Returns:

        """
        from dc_model_repo.base.mr_log import logger

        # 1. 参数校验
        if calc_all_proba is True and calc_max_proba is False:
            logger.warning("已经设置calc_all_proba为True, 忽略calc_max_proba为False。")
            calc_max_proba = True

        # 2. 预测并追加预测结果
        output = self.outputs[0]
        prediction = self.model.predict(X)

        # 3. 计算概率
        proba = None
        if calc_max_proba:  # 只要满足任何一个条件就计算
            # 2.1. 有predict_proba但是不一定调用成功。voting 算法，22.x版本以前为hard时 此方法不能用，如果不成功填充NaN空值。
            if output.max_prob_field is not None:
                try:
                    proba = self.model.predict_proba(X)  # 如果概率列不为空，就一定有概率方法，减少重复判断
                    from deeptables.utils import consts
                    if self.model.task == consts.TASK_BINARY:
                        # 对于二分类 dt给出的概率就是第二个label的概率
                        proba = proba.reshape((1, -1))[0]
                        proba = pd.DataFrame(data={0: (1 - proba), 1: proba}).values  # dt输出的是第2个label的概率
                except Exception as e:
                    logger.warning("调用predict_proba方法失败，设置概率输出值为NaN", e)
                    proba = None
            else:
                logger.warning("设置了calc_max_proba=True，但是Step的Output中没有最大概率列，跳过计算。")

        # 写一个方法，把概率给进去，输出给进去，自动组装
        X = operator_output_util.make_predict_output_data_frame(X, prediction, proba, calc_all_proba, output)

        return X

    def predict(self, X, calc_max_proba=True, **kwargs):
        return self.transform(X, calc_max_proba=calc_max_proba, **kwargs)

    def persist_explanation(self, fs, destination):
        from dc_model_repo.base.mr_log import logger
        # 1. 落地后的目录找到h5的文件名
        h5_file = None
        dt_model_dir = os.path.join(destination, self.model_path)
        logger.info("在目录[{}]查找h5文件。".format(dt_model_dir))
        for f in fs.listdir(dt_model_dir):
            if f[-3:] == '.h5':
                h5_file = f
        if h5_file is None:
            raise Exception("DeepTables 模型没有生成h5文件。")

        # 2. 生成可视化信息
        cd = ChartData(name='keras',
                       type='netron',
                       data=None,
                       attachments={"path": "data/model/{}".format(h5_file)})
        explanation = [cd]

        # 3. 写入数据
        self.persist_explanation_object(fs, destination, explanation)

    def get_outputs(self, x, y=None, options=None, **kwargs):
        from dc_model_repo.base.mr_log import logger
        # 1. 解析输出列的名称和类型
        output_data_type = self.get_as_pd_data_type(y)
        output_name = self.output_cols[0]  # 用户设置的输出列名称

        # 2. 处理概率列
        labels = None
        from deeptables.utils import consts
        if hasattr(self.model, 'predict_proba') and self.model.task in [consts.TASK_BINARY, consts.TASK_MULTICLASS]:  # 如果有概率方法并且是分类问题就生成概率列
            labels = self.model.preprocessor.labels
        else:
            logger.info("模型=%s没有predict_proba方法或者不是分类模型，不生成概率列。" % str(self.model))

        output = operator_output_util.make_output(output_name, output_data_type, labels, 'float64')
        return [output]

    def get_params(self):
        model_config = self.operator.config
        # 通过dir的方式获取所有属性
        params = []
        for attr_name in dir(model_config):
            attr = getattr(model_config, attr_name)
            # 过滤private，protected 属性
            if not attr_name.startswith("_"):
                # 过滤方法
                if not callable(attr):
                    params.append(Param(name=attr_name, type='str', value=str(attr)))

        params.append(Param('input_cols', type='str', value=str(self.input_cols)))
        return params

    def fit_model(self, X, y=None, options=None, **kwargs):
        """训练DT网络。

        Args:
            X:
            y:
            options(dict): 送给模块的参数, 请参考 deeptables.models.deeptable.DeepTable.fit，还可以填写：
              - validation_split_strategy(str): 拆分策略，默认为random_split，可选random_split和k-fold，
                  当为k-fold时调用df.fit_cross_validation，否则调用dt.fit。
            **kwargs:

        Returns:

        """
        validation_split_strategy = options.get("validation_split_strategy", 'random_split')

        if "validation_split_strategy" in options:
            del options["validation_split_strategy"]  # 删除这个key

        # 训练模型
        if validation_split_strategy == 'random_split':
            model, history = self.operator.fit(X, y, **options)
        elif validation_split_strategy == 'k-fold':
            oof_proba, eval_proba, test_proba = self.operator.fit_cross_validation(X, y, **options)
        else:
            raise Exception("validation_split_strategy仅支持random_split和k-fold，当前输入为：%s" % validation_split_strategy)

        return self.operator

    def get_persist_step_ignore_variables(self):
        return ['operator', 'model']

    def get_targets(self, X, y=None, options=None, **kwargs):
        target_name = self.target_cols[0]
        output_field_type = self.get_as_pd_data_type(y)
        return [Field(target_name, output_field_type)]

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


class DeepTablesDCStep(DeepTablesDCEstimator):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("DeepTablesDCStep has been renamed to DeepTablesDCEstimator, and will be removed later. Please use DeepTablesDCEstimator instead.", category=DeprecationWarning)
        super(DeepTablesDCStep, self).__init__(*args, **kwargs)
