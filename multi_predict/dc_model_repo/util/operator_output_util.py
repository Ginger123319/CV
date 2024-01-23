# -*- encoding: utf-8 -*-
"""
Pipeline/Step 输出字段命名工具。
"""

import numpy as np

from dc_model_repo.base import Field, Output
import pandas as pd

from dc_model_repo.util import validate_util


def get_label_prob_fields(prefix, classes_, type_name):
    """生成每个label的概率列。
    Args:
        prefix:
        classes_:
        type_name:
    Returns:
    """
    return [Field('%s_proba_%s' % (prefix, c), type_name) for c in classes_]


def get_max_prob_field(prefix, type_name):
    return Field("%s_max_proba" % prefix, type_name)


def get_raw_prob_field(prefix, type_name):
    return Field("%s_raw_proba" % prefix, type_name)


def __fill_with_na__(input_df, field):
    if field is not None:
        input_df.loc[:, field.name] = np.NaN
    return input_df


def make_predict_output_data_frame(input_df, prediction, proba, calc_all_proba, output, preserve_origin_cols=False,
                                   learning_type=None, positive_label=None, labels=None, binary_threshold=None):
    """解析概率数据生成DataFrame。
        1. 把结果数据拼接到input_df中
        2. 把概率数拼接到input_df中，保证output中有的列，输出的DataFrame中都有
            1. 如果有概率就拼接
            2. 如果没有概率就填充NaN

        概率列为空的情况：
            1. 模型本身没有概率
            2. 关闭了概率计算
        排查方法：
            1. 检查meta.json 中是否有概率输出
            2. 查看调用方法是否开启了概率计算

        概率列可能为空，但是只要Ouput中有就一定不会少。

    Args:
        input_df: 输入的DataFrame
        prediction: 预测结果，ndarray
        proba: 概率数据，shape=(n, n_classes)，可以为空，如果为空，所有的概率信息将使用NaN填充，实际的输出结果以output为准。
        preserve_origin_cols: 是否保留原始列。默认False，不保留。
        learning_type:
        positive_label:
        labels:
        binary_threshold:

    Returns:
        返回DataFrame
    """
    from dc_model_repo.base.mr_log import logger
    # 1. 合并预测结果
    # input_df[output.name] = prediction  # fix SettingWithCopyWarning
    if preserve_origin_cols:
        output_df = input_df
        output_df.loc[:, output.name] = prediction
    else:
        output_df = pd.DataFrame(prediction, columns=[output.name])

    # 2. 合并概率结果
    if proba is not None:
        # 2.1. 确保为ndarray
        if not isinstance(proba, np.ndarray):
            proba = np.array(proba)

        # 2.2. 验证概率的shape正确
        if output.label_prob_fields is not None:
            if len(output.label_prob_fields) != proba.shape[1]:
                raise Exception("参数label_proba_name中标签的个数和概率的个数不相同。")

        # 2.3. 生成最大概率列
        if output.max_prob_field is not None:  # 兼容 max_prob_field 为空
            output_df.loc[:, output.max_prob_field.name] = np.amax(proba, axis=1)

        # 2.4. 生成标签概率列
        prob_cols = [c.name for c in output.label_prob_fields]
        if calc_all_proba:
            if validate_util.is_non_empty_list(output.label_prob_fields):
                for i, n in enumerate(output.label_prob_fields):
                    output_df[n.name] = proba[:, i]
        else:
            # 填充空
            if validate_util.is_non_empty_list(output.label_prob_fields):
                for f in output.label_prob_fields:
                    output_df = __fill_with_na__(output_df, f)

        # 2.5. 生成原始概率
        if calc_all_proba:
            if output.raw_prob_field is not None:
                output_df[output.raw_prob_field.name] = proba.tolist()  # df里面只能有普通数组
        else:
            output_df = __fill_with_na__(output_df, output.raw_prob_field)

        # 2.6. 处理二分类阈值
        from dc_model_repo.base import LearningType
        if learning_type == LearningType.BinaryClassify and positive_label is not None and binary_threshold is not None:
            positive_col = get_label_prob_fields(output.name, [positive_label], "float64")[0].name
            if positive_col not in prob_cols:
                logger.info("没有在概率列{}中找到正样本[{}]对应的列[{}],不使用阈值[{}]".format(prob_cols, positive_label, positive_col, binary_threshold))
            else:
                negative_filter = [str(l) for l in labels if str(l) != positive_label]
                if len(negative_filter) == 1:
                    negative_label = negative_filter[0]
                else:
                    negative_col = [c for c in prob_cols if c != positive_col][0]
                    negative_label = negative_col[len("{}_proba_".format(output.name))]

                positive_mask = output_df[positive_col] >= binary_threshold
                output_df.loc[positive_mask, output.name] = positive_label
                output_df.loc[~positive_mask, output.name] = negative_label
                if output.max_prob_field.name in output_df.columns:
                    output_df.loc[positive_mask, output.max_prob_field.name] = output_df.loc[positive_mask, positive_col]
                    output_df.loc[~positive_mask, output.max_prob_field.name] = 1 - output_df.loc[~positive_mask, positive_col]
                if output_df[output.name].dtype.name != output.type:
                    try:
                        output_df[output.name] = output_df[output.name].astype(output.type)
                    except:
                        logger.info("尝试将预测列类型[{}]转为原始数据类型[{}]失败！".format(output_df[output.name].dtype.name, output.type))

    else:
        # 如果没有概率，全部填充NaN
        output_df = __fill_with_na__(output_df, output.max_prob_field)
        output_df = __fill_with_na__(output_df, output.raw_prob_field)
        if validate_util.is_non_empty_list(output.label_prob_fields):
            for f in output.label_prob_fields:
                output_df = __fill_with_na__(output_df, f)

    return output_df


def make_output(output_name, output_data_type, classes_=None, proba_data_type=None):
    """组装Output对象。 通常分类模型都会有概率，也都会有classes_，对应的会有最大概率，原始概率和每个标签的概率，
       所以，只要有概率输出，所有的概率都会有，否则都没；

    Args:
        output_name:
        output_data_type:
        proba_data_type:
        calc_all_proba:
        classes_: 分类模型的标签，只有标签不为空才会生成 原始概率列，和每个标签的概率列。

    Returns:

    """

    # 1. 计算概率
    max_prob_field = None
    label_prob_fields = None
    raw_prob_field = None

    if validate_util.is_non_empty_list(classes_):
        # 1.1. 生成最大概率
        max_prob_field = get_max_prob_field(output_name, proba_data_type)
        # 1.2. 生成标签概率
        label_prob_fields = get_label_prob_fields(output_name, classes_, proba_data_type)
        # 1.3. 生成原始概率列 2021.6.29:注释掉这里，不生成raw_proba列，有点冗余
        # raw_prob_field = get_raw_prob_field(output_name, proba_data_type)
        # raw_prob_field.shape = [1, len(classes_)]

    output = Output(name=output_name,
                    type=output_data_type,
                    max_prob_field=max_prob_field,
                    raw_prob_field=raw_prob_field,
                    label_prob_fields=label_prob_fields)
    return output
