# -*- encoding: utf-8 -*-
import abc
import numpy as np
import pandas as pd
import six

from dc_model_repo.base import DatasetType
from dc_model_repo.util import validate_util
from dc_model_repo.base.mr_log import logger


def dask_replace_col(input_ddf, col, fn):
    import dask.dataframe as ddf
    assert isinstance(input_ddf, ddf.DataFrame)
    cols = list(input_ddf.columns)
    cur_col = input_ddf[col]
    base_df = input_ddf[[c for c in cols if c!=col]]
    to_merge = fn(cur_col).rename(col)
    to_merge = to_merge.to_frame()
    return base_df.merge(right=to_merge, left_index=True, right_index=True)[cols]

@six.add_metaclass(abc.ABCMeta)
class TypeConverter(object):
    """类型转换逻辑抽象。抽象出来一层，仅关注转换的条件和转换的内容。
    """

    def __init__(self, supported_input_type, convert_output_types):
        self.supported_input_type = supported_input_type  # 当前转换器支持的类型。
        self.convert_output_types = convert_output_types  # 转换成的目标列

    def is_compatible(self, input_col_type, output_col_type):
        """测试当前转换器是否可以转换数据，要求输入和输出在当前转换器的支持范围内。
        Returns:
            返回bool类型。
        """
        # if input_col_type == self.supported_input_type:
        #     return output_col_type in self.convert_output_types

        # Fixed by zk. 2020.10.19. 当输入列类型input_col_type不为supported_input_type时，会返回None
        return (input_col_type == self.supported_input_type) and (output_col_type in self.convert_output_types)

    @abc.abstractmethod
    def convert_data(self, df, col_name, input_col_type, feature_col_type):
        """对数据列进行转换。
        Args:
            df: 数据。
            col_name: 列名。
            input_col_type: 输入特征列类型。
            feature_col_type: 模型特征列类型。
        Returns:
            转换后的df。
        """
        raise NotImplemented

    def convert(self, df, col_name, input_col_type, output_col_type):
        if self.is_compatible(input_col_type, output_col_type):
            return self.convert_data(df, col_name, input_col_type, output_col_type)
        else:
            raise Exception("当前转换器无法处理类型:%s" % input_col_type)


@six.add_metaclass(abc.ABCMeta)
class SparkConverter(TypeConverter):
    @staticmethod
    def cast_type(df, col_name, dest_type):
        # Modified by zk. 2020.10.19. 保持列的顺序不变
        select_expr_args = []
        for c in df.columns:
            cur_exp = "cast(`{}` as {}) as `{}`".format(c, dest_type, col_name) if c == col_name else "`{}`".format(c)
            select_expr_args.append(cur_exp)
        df = df.selectExpr(select_expr_args)
        return df


class SparkDFIntegerConverter(SparkConverter):
    """
    int 根据模型输入要求可以转换为float/double。
    """

    def __init__(self):
        super(SparkDFIntegerConverter, self).__init__("integer", ["float", "double"])

    def convert_data(self, df, col_name, input_col_type, feature_col_type):
        dest_types = ["float", "double"]
        if feature_col_type in dest_types:
            return self.cast_type(df, col_name, feature_col_type)
        else:
            raise Exception("类型%s仅能转换为%s" % (input_col_type, ",".join(dest_types)))


class SparkDFLongConverter(SparkConverter):
    """
    long 根据模型输入要求可以转换为integer/float/double。
    """

    def __init__(self):
        super(SparkDFLongConverter, self).__init__("long", ["integer", "float", "double"])

    def convert_data(self, df, col_name, input_col_type, feature_col_type):
        if feature_col_type in self.convert_output_types:
            return self.cast_type(df, col_name, feature_col_type)
        else:
            raise Exception("类型%s仅能转换为%s" % (input_col_type, ",".join(self.convert_output_types)))


class SparkDFFloatConverter(SparkConverter):
    """
    float 可以转换为 double。
    """

    def __init__(self):
        super(SparkDFFloatConverter, self).__init__("float", ["double"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        return self.cast_type(df, col_name, output_col_type)


class SparkDFDoubleConverter(SparkConverter):
    """
    double可以容忍float 。
    """

    def __init__(self):
        super(SparkDFDoubleConverter, self).__init__("double", ["float"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        from dc_model_repo.base.mr_log import logger
        logger.warning("当前列名%s类型为%s要求转换为%s，为降低精度，不予转换。" % (col_name, self.supported_input_type, output_col_type))
        return df


class PandasDFInt32Converter(TypeConverter):
    """
     int32可转int64/float32/float64。
    """

    def __init__(self):
        super(PandasDFInt32Converter, self).__init__("int32", ["int64", "float32", "float64"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        if isinstance(df, pd.DataFrame):
            df[[col_name]] = df[[col_name]].astype(output_col_type)
        else:
            df = dask_replace_col(df, col_name, lambda s: s.astype(output_col_type))
        return df


class PandasDFInt64Converter(TypeConverter):
    """
     int64可转float32/float64;容忍要求转为int32但是不转换。
    """

    def __init__(self):
        super(PandasDFInt64Converter, self).__init__("int64", ["int32", "float32", "float64"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        if output_col_type == "int32":
            from dc_model_repo.base.mr_log import logger
            logger.warning("当前列名%s类型为%s要求转换为%s，为避免降低精度，不予转换。" % (col_name, self.supported_input_type, output_col_type))
        else:
            if isinstance(df, pd.DataFrame):
                df[[col_name]] = df[[col_name]].astype(output_col_type)
            else:
                df = dask_replace_col(df, col_name, lambda s: s.astype(output_col_type))
        return df


class PandasDFFloat32Converter(TypeConverter):
    """
    float32可转float64
    """

    def __init__(self):
        super(PandasDFFloat32Converter, self).__init__("float32", ["float64"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        df[[col_name]] = df[[col_name]].astype(output_col_type)
        if isinstance(df, pd.DataFrame):
            df[[col_name]] = df[[col_name]].astype(output_col_type)
        else:
            df = dask_replace_col(df, col_name, lambda s: s.astype(output_col_type))
        return df


class PandasDFFloat64Converter(TypeConverter):
    """
    float64容忍转换为float32，但是不予转换。
    """

    def __init__(self):
        super(PandasDFFloat64Converter, self).__init__("float64", ["float32"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        from dc_model_repo.base.mr_log import logger
        logger.warning("当前列名%s类型为%s要求转换为%s，为避免降低精度，不予转换。" % (col_name, self.supported_input_type, output_col_type))
        return df


class PandasDFObjectConverter(TypeConverter):
    """
    字符串可转日期。
    """

    def __init__(self):
        super(PandasDFObjectConverter, self).__init__("object", ["datetime64[ns]"])

    def convert_data(self, df, col_name, input_col_type, output_col_type):
        if isinstance(df, pd.DataFrame):
            df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        else:
            from dask import dataframe as ddf
            df = dask_replace_col(df, col_name, lambda s: ddf.to_datetime(s))
        return df


# 默认转换器
Spark_DF_Converter_List = (SparkDFIntegerConverter(), SparkDFFloatConverter(), SparkDFDoubleConverter(), SparkDFLongConverter())

Pandas_DF_Converter_List = (PandasDFInt32Converter(), PandasDFInt64Converter(), PandasDFFloat32Converter(), PandasDFFloat64Converter(), PandasDFObjectConverter())


def cast_df_data_type(df, schema, input_data_name_type_dict, converter_list):
    # 1. 定义获取转换器方法
    def get_converter(input_col_type, feature_col_type):
        for converter_item in converter_list:
            if converter_item.is_compatible(input_col_type, feature_col_type):
                return converter_item
        return None

    # 2. 遍历输入，进行类型转换。
    for f in schema:
        feature_name = f.name
        # 2.1. 输入列缺失
        if feature_name not in input_data_name_type_dict:
            raise Exception("需要输入字段%s, 输入数据中没有提供。" % feature_name)

        data_type_name = input_data_name_type_dict[feature_name]
        feature_type_name = f.type

        # 2.2. 类型不符，尝试转换
        if feature_type_name.lower() != data_type_name.lower():  # 处理类型 pd.Int32Dtype()
            converter = get_converter(data_type_name, feature_type_name)
            if converter is not None:
                logger.info("Convert dtype for [{}]: {} --> {}".format(feature_name, data_type_name, feature_type_name))
                df = converter.convert(df, feature_name, data_type_name, feature_type_name)
            else:
                msg = "没有合适转换器，将类型%s转换为%s，列名=%s。跳过该转换！" % (data_type_name, feature_type_name, feature_name)
                logger.error(msg=msg)

    return df


def cast_spark_df_data_type(df, input_features, rules):
    """对Spark DataFrame 在类型不匹配时进行类型转换，转换规则：
        1. int 可转 float/double
        2. float 可转 double
        3. 需要float实际为double时警告
    Args:
        df:
        input_features: 模型的输入特征。
        cut_unnecessary_cols: 是否删除不相关的列
        rules:
    Returns:
    转换处理后的DataFrame。
    """
    input_data_name_type_dict = {}
    for field in df.schema:
        input_data_name_type_dict[field.name] = field.dataType.typeName()
    df = cast_df_data_type(df, input_features, input_data_name_type_dict, rules)
    return df


def cast_pandas_df_data_type(df, schema, cut_unnecessary_cols, rules):
    """对Pandas DataFrame 在类型不匹配时进行类型转换，默认转换规则：
         1. int32可转int64，反过来警告；还可以转float32/float64。
         2. int64 可转float32/float64
         3. float32可转 float64，反过来警告。
         4. 字符串可转日期。
         5. 除了以上转换规则外报错。
    Args:
        df: 需要校验、转换的Pandas DataFrame
        schema(dict):
        cut_unnecessary_cols:
        rules:
    Returns:
    """
    from dc_model_repo.base.mr_log import logger

    dtypes_dict = df.dtypes.to_dict()
    input_data_name_type_dict = {}
    for c in dtypes_dict:
        input_data_name_type_dict[c] = dtypes_dict[c].name.lower()  # 处理pd.Int32Dtype() 类型
    df = cast_df_data_type(df, schema, input_data_name_type_dict, rules)

    if cut_unnecessary_cols is True:
        # feature_names = set([f.name for f in schema])  # 模型中需要的列
        # 不能使用set，会导致列的顺序乱掉
        feature_names = [f.name for f in schema]  # 模型中需要的列

        # unnecessary_cols = set(dtypes_dict.keys()) - feature_names
        # 使用list求差集，不能使用set
        unnecessary_cols = [i for i in dtypes_dict.keys() if i not in feature_names]
        if validate_util.is_non_empty_list(unnecessary_cols):
            logger.warning("当前输入多余的列: %s", ",".join(list(unnecessary_cols)))

        return df[list(feature_names)]
    else:
        return df


def cast_dask_df_data_type(df, schema, cut_unnecessary_cols=False, rules=None):
    from dask import dataframe as dd
    if isinstance(df, pd.DataFrame):
        df = dd.from_pandas(data=df, chunksize=10000)

    # dtypes_dict = df.dtypes.to_dict()
    # input_data_name_type_dict = {}
    # for c in dtypes_dict:
    #     input_data_name_type_dict[c] = dtypes_dict[c].name.lower()  # 处理pd.Int32Dtype() 类型
    # df = cast_df_data_type(df, schema, input_data_name_type_dict, rules)

    df = cast_pandas_df_data_type(df=df, schema=schema, cut_unnecessary_cols=cut_unnecessary_cols, rules=rules)

    return df


def validate_and_cast_input_data(input_data, dataset_type, schema, remove_unnecessary_cols=False, rules=None):
    """校验输入数据是否与训练的格式一致，包括：
       1. 检查输入的字段的名称和类型与模型需要一致，如果不对应将进行转换，转换的原则为仅关心那些可以转换的类型。
       2. 根据需要去除多余的列。
    Args:
        input_data (Union(pandas.DataFrame, pyspark.sql.DataFrame, numpy.ndarray)): 输入数据。
        dataset_type: 数据集类型，可选值见: `dc_model_repo.base.DatasetType`。
        schema(list(Field)): 数据格式。
        remove_unnecessary_cols (bool): 去除不在schema中的列。
        rules: 转换规则，可以自定义转换规则。
    Raises:
        Exception: 格式与训练所用格式不匹配。
    """

    # 1. 检查pandas类型数据
    from dc_model_repo.base.data_sampler import NumpyDataSampler, DictDataSampler, ListDataSampler, DaskDataFrameDataSampler
    from dc_model_repo.base import Field

    if dataset_type == DatasetType.PandasDataFrame:
        # 1.1. 校验输入类型
        if not isinstance(input_data, pd.DataFrame):
            raise Exception("输入数据不是pd.DataFrame, 它是：%s" % str(type(input_data)))

        # 1.2. 校验schema
        if rules is None:
            rules = Pandas_DF_Converter_List
        return cast_pandas_df_data_type(input_data, schema, remove_unnecessary_cols, rules)

    # 2. 检查spark dataframe 类型数据
    elif dataset_type == DatasetType.PySparkDataFrame:
        from pyspark import sql
        if not isinstance(input_data, sql.DataFrame):
            raise Exception("输入数据不是pyspark.sql.dataframe.Dataframe, 无法提取样本数据, 它是：%s" % str(type(input_data)))
        if rules is None:
            rules = Spark_DF_Converter_List
        return cast_spark_df_data_type(input_data, schema, rules)

    elif dataset_type == DatasetType.ArrayData:
        if isinstance(input_data, np.ndarray) or isinstance(input_data, list) or isinstance(input_data, pd.DataFrame):
            pass
        else:
            raise Exception("输入数据不是 'arrayData' 数据, 它是: %s" % str(input_data))
        return input_data
    elif dataset_type == NumpyDataSampler.DATA_TYPE:
        if not NumpyDataSampler.is_compatible(input_data):
            raise Exception("输入数据[{}.{}]与训练时的{}不一致，请检查数据类型！".format(type(input_data).__module__, type(input_data).__name__, dataset_type))
        # 检查数据形状
        ele_type, ele_shape = schema[0].type, schema[0].shape
        assert all([a == b for a, b in zip(ele_shape[1:], input_data.shape[1:])]), "输入数据的形状{}与训练时{}不一致".format(str(input_data.shape[1:]), str(ele_shape[1:]))
        # 进行必要的转换
        if input_data.dtype.name != ele_type:
            input_data = input_data.astype(ele_type)
        return input_data
    elif dataset_type == DictDataSampler.DATA_TYPE:
        if not DictDataSampler.is_compatible(input_data):
            raise Exception("输入数据[{}.{}]与训练时的{}不一致，请检查数据类型（如果确实传了dict数据，留意k-v对中value是否提供了不兼容的类型）！".format(type(input_data).__module__, type(input_data).__name__, dataset_type))
        # 检查对应项是否一致
        schema_dict = {f.name: f for f in schema}
        cols_lack = []
        cols_diff_type = []
        data_filtered = {}
        for k, v in schema_dict.items():
            if k in input_data:
                cur_type = "{}.{}".format(type(input_data[k]).__module__, type(input_data[k]).__name__)
                if isinstance(input_data[k], np.ndarray):
                    if v.struct is not None and v.struct != Field.STRUCT_NDARRAY:
                        cols_diff_type.append((k, (cur_type, v.struct)))
                    else:
                        data_filtered[k] = input_data[k]
                else:
                    if v.type != type(input_data[k]).__name__:
                        cols_diff_type.append((k, (cur_type, v.type)))
                    else:
                        data_filtered[k] = input_data[k]
            else:
                cols_lack.append(k)
        msg = ""
        if len(cols_lack) > 0:
            msg += "输入数据较训练时缺少键：{} ".format(str(cols_lack))
        if len(cols_diff_type) > 0:
            for k, (cur_type, pre_type) in cols_diff_type:
                msg += "\n键[{}]对应的值类型[{}]与之前训练时[{}]不一致".format(repr(k), cur_type, pre_type)
        if msg != "":
            raise Exception("数据不匹配，请按如下提示检查数据：\n{}".format(msg))
        return data_filtered if remove_unnecessary_cols else input_data
    elif dataset_type == ListDataSampler.DATA_TYPE:
        if not ListDataSampler.is_compatible(input_data):
            raise Exception("输入数据[{}.{}]与训练时的{}不一致，请检查数据类型！".format(type(input_data).__module__, type(input_data).__name__, dataset_type))
        assert len(input_data) == len(schema), "输入列表长度[{}]与之前训练时[{}]不一致！请检查数据！".format(len(input_data), len(schema))
        schema_dict = {f.name: f for f in schema}
        cols_diff_type = []
        for i, v in enumerate(input_data):
            cur_type = "{}.{}".format(type(v).__module__, type(v).__name__)
            if cur_type != schema_dict[i].type:
                cols_diff_type.append((i, (cur_type, schema_dict[i].type)))
        if len(cols_diff_type) > 0:
            msg = "当前输入数据的类型与训练时不一致(以下索引从0计数)："
            for i, (cur_type, pre_type) in cols_diff_type:
                msg += "\n索引为{}的位置，当前数据类型为[{}]，而之前为[{}]".format(str(i), cur_type, pre_type)
            raise Exception(msg)
        return input_data
    elif dataset_type == DaskDataFrameDataSampler.DATA_TYPE:

        if rules is None:
            rules = Pandas_DF_Converter_List

        if not DaskDataFrameDataSampler.is_compatible(input_data):
            raise Exception("输入的数据类型[{},{}]与训练时的{}不兼容，目前只兼容DaskDataFrame和PandasDataFrame，请检查数据".format(type(input_data).__module__, type(input_data).__name__, dataset_type))
        input_data = cast_dask_df_data_type(df=input_data, schema=schema, cut_unnecessary_cols=remove_unnecessary_cols, rules=rules)
        return input_data
    else:
        raise Exception("不支持的数据集类型: %s" % dataset_type)


def write_missing_info(df, output_path, args=None):
    """计算缺失比例。供pipes调用。

    Args:
        df: pandas DataFrame，要计算缺失值的数据
        output_path: string，Pipes设置的输出路径，绝对路径
        args: dict，额外参数。其中"cache_sub_path"可不传，使用output_path跟cache_sub_path拼接在一起来获取写文件的路径。
    """
    if args is None:
        args = {'cache_sub_path': '.cache//all//column.stat'}
    assert isinstance(df, pd.DataFrame)
    logger.info("开始计算数据集缺失率")
    logger.info("Dataset shape: {}".format(df.shape))
    col_cnt = df.columns.value_counts()
    col_redundant = col_cnt[col_cnt > 1]
    assert len(col_redundant)==0, "数据中存在重复的列：{}".format(list(col_redundant.items()))
    logger.info("output_path: {}".format(output_path))
    logger.info("args: {}".format(args))
    assert args is not None
    suffix = args.get("cache_sub_path")
    assert suffix is not None
    from pathlib import Path
    output_path = Path(output_path).absolute()
    write_path = str(output_path) + suffix
    logger.info("目标文件: {}".format(write_path))
    write_path = Path(write_path)
    logger.info("Write path: {}".format(write_path))
    if not write_path.parent.exists() or write_path.parent.is_file():
        write_path.parent.mkdir(parents=True, exist_ok=True)

    col_info = list()
    import time
    timestamp = int(time.time()*1000)
    row_cnt = df.shape[0]
    na_cnt = row_cnt - df.count(axis=0)
    na_ratio = na_cnt / row_cnt
    for c, d in df.dtypes.items():
        if d.name.startswith("int") or d.name.startswith("float"):
            col_info.append(
                {"numerical": {
                    "qulity": {
                        "indicator": 0,
                        "bars": [{
                            "x": "missing",
                            "y": na_ratio[c]}]}},
                    "colName": c,
                    "timestamp": timestamp
                })
        else:
            col_info.append(
                {"categorical": {"qulity": {"missing": na_ratio[c]}},
                 "colName": c,
                 "timestamp": timestamp
                 })
    col_info = {"result": col_info}
    import json
    with open(write_path, "w", encoding="utf-8") as f:
        logger.info("写文件……")
        logger.info(col_info)
        json.dump(col_info, f)
    logger.info("计算数据集缺失率结束")


def write_shape_info(df, output_path, args=None):
    """计算数据行列count。供pipes调用。

    Args:
        df: pandas DataFrame，要计算行列count的数据
        output_path: string，Pipes设置的输出路径，绝对路径
        args: dict，额外参数。其中"cache_sub_path"可不传，使用output_path跟cache_sub_path拼接在一起来获取写文件的路径。

    {
        "row_count": 10,
        "column_count": 10
    }
    """
    if args is None:
        args = {'cache_sub_path': '.cache//all//shape.summary'}
    assert isinstance(df, pd.DataFrame)
    logger.info("开始计算数据集shape")
    logger.info("Dataset shape: {}".format(df.shape))
    logger.info("output_path: {}".format(output_path))
    logger.info("args: {}".format(args))
    assert args is not None
    suffix = args.get("cache_sub_path")
    assert suffix is not None
    from pathlib import Path
    output_path = Path(output_path).absolute()
    write_path = str(output_path) + suffix
    logger.info("目标文件: {}".format(write_path))
    write_path = Path(write_path)
    logger.info("Write path: {}".format(write_path))
    if not write_path.parent.exists() or write_path.parent.is_file():
        write_path.parent.mkdir(parents=True, exist_ok=True)

    shape_info = {'row_count': df.shape[0], 'column_count': df.shape[1]}

    import json
    with open(write_path, "w", encoding="utf-8") as f:
        logger.info("写文件……")
        logger.info(shape_info)
        json.dump(shape_info, f)
    logger.info("计算数据集shape结束")


if __name__ == "__main__":
    # from dc_model_repo.base import Field
    #
    # # 测试numpy类型的转换
    # np_input = np.ones((2, 3), dtype="int64")
    # np_type = "NumpyArray"
    # np_schema = [Field("NumpyArray", "float32", (1, 3))]
    # output_data = validate_and_cast_input_data(np_input, np_type, np_schema)
    # print(output_data.dtype.name, "\n", output_data)
    #
    # # 测试dict类型的转换
    # dict_input = {"numpy": np.ones((2, 3), dtype="int64"),
    #               "pandas": pd.DataFrame(np.ones((5, 6))),
    #               "int": 1,
    #               "string": "hello",
    #               "float": 1.1,
    #               "more": "0"}
    # dict_type = "Dict"
    # dict_schema = [Field("numpy", "numpy.ndarray", (1, 3)),
    #                Field("pandas", "pandas.core.frame.DataFrame", (1, 3)),
    #                Field("int", "builtins.int", None),
    #                Field("string", "builtins.str", None),
    #                Field("float", "builtins.float", None)]
    # output_data = validate_and_cast_input_data(dict_input, dict_type, dict_schema, remove_unnecessary_cols=True)
    # print(output_data.keys())
    #
    # # 测试list类型的转换
    # list_input = [np.ones((2, 3), dtype="int64"), pd.DataFrame(np.ones((5, 6))), 1, "hello", 1.1]
    # list_type = "List"
    # list_schema = [Field(0, "numpy.ndarray", (1, 3)),
    #                Field(1, "pandas.core.frame.DataFrame", (1, 3)),
    #                Field(2, "builtins.int", None),
    #                Field(3, "builtins.str", None),
    #                Field(4, "builtins.float", None)]
    # output_data = validate_and_cast_input_data(list_input, list_type, list_schema, remove_unnecessary_cols=True)
    # print(len(output_data))

    # df = pd.read_csv("E:\\workspace\\data_canvas\\dc-sdk-mr-py\\tests\\datasets\\house_prices\\train.csv")
    # # df = pd.concat([df, df["Id"]], axis=1)
    # write_missing_info(df, "./tmp", args={'cache_sub_path': '.cache//all//columns.summary'})

    df = pd.read_csv("/Users/bjde/Desktop/data/二分类/bank_data_small.csv")
    write_shape_info(df, "/tmp/a")

