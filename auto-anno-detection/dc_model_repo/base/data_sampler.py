# -*- encoding: utf-8 -*-
import math

import abc
import numpy as np
import pandas as pd
import six

from dc_model_repo.base import PKLSerializable, Field, DatasetType
from dc_model_repo.util import str_util, json_util
from dc_model_repo.util.decorators import deprecated


@six.add_metaclass(abc.ABCMeta)
class DataSampler(PKLSerializable):
    """样本数据抽样器。

    Args:
        data: 要进行抽样的数据
        limit (int): 取前多少条数据作为样本
    """

    def __init__(self, data, limit=1):
        self.limit = 1
        self.data = data

    @abc.abstractmethod
    def get_features(self):
        """推测样本数据的特征。

        Returns:
            list: Field列表
        """
        pass

    def get_input_features(self, input_cols=None):
        """根据input_cols获取输入列。

        结果是get_features的子集，如果input_cols中的一些列是数据中没有的，要报错。

        Args:
            input_cols: 输入列

        Returns: 输入列对应的features
        """
        feature_dict = {c.name: c for c in self.get_features()}
        if input_cols is None:
            return self.get_features()
        input_features = []
        cols_lack = []
        for c in input_cols:
            if c in feature_dict:
                input_features.append(feature_dict[c])
            else:
                cols_lack.append(c)
        if len(cols_lack) > 0:
            cols_lack = [repr(c) for c in cols_lack]
            raise ValueError("输入数据与预期不一致：输入列中指定有[{}], 但是数据中却没有，请检查输入列！".format(", ".join(cols_lack)))

        return input_features

    @staticmethod
    def get_data_type():
        """
        Returns:
            str: 抽样器处理的数据类型。
        """
        pass

    @abc.abstractmethod
    def sample_data_list(self):
        """抽样获取样本数据。

        Returns:
            list: 返回数据。
        """
        pass

    @staticmethod
    def is_compatible(data):
        """判断数据能否被当前抽样器处理。

        Args:
            data: 输入数据

        Returns:
            bool: 为true表示支持当前数据格式。
        """
        pass

    def to_json_str(self):
        data_list = self.sample_data_list()
        self.replace2None(data_list)
        return json_util.to_json_str(data_list)

    def replace2None(self, array):
        """替换array中inf,nan为null。 为inplace操作。

        Args:
            array: 输入数据
        Returns:
            替换后的数据

        """

        def is_invalided(v):
            try:
                return math.isnan(v) or math.isinf(v)
            except Exception as e:
                return False

        for i, value in enumerate(array):
            if isinstance(value, list):
                self.replace2None(value)
            else:
                if value is not None:
                    if is_invalided(value):
                        array[i] = None


class PandasDataFrameDataSampler(DataSampler):
    """Pandas DataFrame数据抽样"""

    DATA_TYPE = DatasetType.PandasDataFrame

    @staticmethod
    def is_compatible(data):
        return isinstance(data, pd.DataFrame)

    def __init__(self, data, limit=1):
        super(PandasDataFrameDataSampler, self).__init__(data, limit)
        self.data = data[:limit]
        if not self.is_compatible(data):
            raise Exception("'df' 不是pd.DataFrame, 无法提取样本数据, 它是：%s" % str(type(data)))

    def sample_data_list(self):
        sample_data = self.data

        # 把 pandas datetime64 类型的数据转换成python原生的日期类型。
        for s in self.get_features():
            if s.type.startswith('datetime64'):
                col_name = s.name
                sample_data[col_name] = pd.DatetimeIndex(sample_data[col_name]).to_native_types()
        # 替换空数据
        # sample_data_list = sample_data.fillna("NaN").replace(np.inf, "Infinity").replace(-np.inf, "-Infinity").values.tolist()
        sample_data_list = sample_data.values.tolist()
        return sample_data_list
        # raise Exception("把数据转换成JSON仅支持np.ndarray和pd.DataFrame .")

    @staticmethod
    def get_data_type():
        return PandasDataFrameDataSampler.DATA_TYPE

    def get_features(self):
        schema = self.data.dtypes
        # 类型名称转换为小写存储, 见https://gitlab.datacanvas.com/APS/dc-sdk-mr-py/issues/147
        return [Field(k, schema[k].name.lower()) for k in schema.keys()]


class DaskDataFrameDataSampler(DataSampler):
    """Dask DataFrame 数据抽样"""
    DATA_TYPE = DatasetType.DaskDataFrame

    @staticmethod
    def is_compatible(data):
        import dask.dataframe as dd
        return isinstance(data, (dd.DataFrame, pd.DataFrame))

    def __init__(self, data, limit=1):
        super(DaskDataFrameDataSampler, self).__init__(data, limit)
        if not self.is_compatible(data):
            raise Exception("'df' 不是dask DataFrame, 无法提取样本数据, 它是：%s" % str(type(data)))
        # head 返回的结果是pandas DataFrame
        import dask.dataframe as dd
        if isinstance(data, dd.DataFrame):
            self.data = data.partitions[0].head(1)
        elif isinstance(data, pd.DataFrame):
            self.data = data.head(1)
        else:
            raise Exception("不支持的类型：{}".format(type(data)))

    def sample_data_list(self):
        sample_data = self.data

        # 把 pandas datetime64 类型的数据转换成python原生的日期类型。
        for s in self.get_features():
            if s.type.startswith('datetime64'):
                col_name = s.name
                sample_data[col_name] = pd.DatetimeIndex(sample_data[col_name]).to_native_types()
        # 替换空数据
        # sample_data_list = sample_data.fillna("NaN").replace(np.inf, "Infinity").replace(-np.inf, "-Infinity").values.tolist()
        sample_data_list = sample_data.values.tolist()
        return sample_data_list
        # raise Exception("把数据转换成JSON仅支持np.ndarray和pd.DataFrame .")

    @staticmethod
    def get_data_type():
        return DaskDataFrameDataSampler.DATA_TYPE

    def get_features(self):
        schema = self.data.dtypes
        # 类型名称转换为小写存储, 见https://gitlab.datacanvas.com/APS/dc-sdk-mr-py/issues/147
        return [Field(k, schema[k].name.lower()) for k in schema.keys()]


class PySparkDataFrameDataSampler(DataSampler):
    DATA_TYPE = DatasetType.PySparkDataFrame

    def __init__(self, data, limit=1):
        # 1. 校验
        if not self.is_compatible(data):
            raise Exception("'df' 不是pyspark.sql.dataframe.Dataframe, 无法提取样本数据, 它是：%s" % str(type(data)))

        # 2. 解析schema
        self.data_features = [Field(str_util.to_str(f.name), str_util.to_str(f.dataType.typeName()))
                              for f in data.schema]

        # 3. 将数据转换为数组存储(避免依赖pandas, 导致上线时与hadoop环境的pandas冲突)
        # data = data.limit(limit).toPandas()
        col_names = data.schema.names
        row_list = data.limit(limit).collect()
        sample_data_list = []
        for row in row_list:
            row_dict = row.asDict()
            values = [row_dict.get(c) for c in col_names]
            sample_data_list.append(values)

        # 5. 调用父类方法
        super(PySparkDataFrameDataSampler, self).__init__(sample_data_list, limit)

    @staticmethod
    def is_compatible(data):
        from pyspark.sql.dataframe import DataFrame
        return isinstance(data, DataFrame)

    def sample_data_list(self):
        return self.data

    @staticmethod
    def get_data_type():
        return PySparkDataFrameDataSampler.DATA_TYPE

    def get_features(self):
        return self.data_features


class ArrayDataSampler(DataSampler):
    DATA_TYPE = DatasetType.ArrayData

    @staticmethod
    def is_compatible(data):
        return isinstance(data, np.ndarray) or isinstance(data, list) or isinstance(data, pd.DataFrame)

    @deprecated(msg="Use NumpyDataSampler or ListDataSampler instead.")
    def __init__(self, data, limit=None):
        super(ArrayDataSampler, self).__init__(data, limit)
        if not self.is_compatible(data):
            raise Exception("'df' 不是np.ndarray, 无法提取样本数据, 它是：%s" % str(type(data)))

        data_dict = {}
        if isinstance(data, np.ndarray):
            data_dict['x0'] = data[:self.limit]
        elif isinstance(data, list):
            for i, d in enumerate(data):
                if isinstance(d, pd.DataFrame):
                    d = d.values
                    d = d[:self.limit]
                elif isinstance(d, list):
                    d = np.array(d)
                    d = d[:self.limit]
                elif isinstance(d, np.ndarray):
                    d = d[:self.limit]
                else:
                    d = np.array(d)
                data_dict['x%d' % i] = d
        elif isinstance(data, pd.DataFrame):  # modelDisstor时多输入的情况，可能是dataframe
            for k in data.keys().values:
                data_dict['%s' % k] = np.array(data[k].values)
        else:
            raise Exception("不支持的数据： %s" % str(type(data)))

        self.data_dict = data_dict

    def sample_data_list(self):
        # deep model sampleData需要多加一层
        return [[self.data_dict[d].tolist() for d in self.data_dict]]

    @staticmethod
    def get_data_type():
        return ArrayDataSampler.DATA_TYPE

    def get_features(self):
        return [Field(d, self.data_dict[d].dtype, self.data_dict[d].shape) for d in self.data_dict]

    def get_input_features(self, input_cols=None):
        return None


class NumpyDataSampler(DataSampler):
    DATA_TYPE = DatasetType.NumpyArray

    @staticmethod
    def is_compatible(data):
        return isinstance(data, np.ndarray) and data.ndim > 1

    def __init__(self, data, limit=1):
        super(NumpyDataSampler, self).__init__(data, limit)
        if not self.is_compatible(data):
            raise Exception("{}不兼容当前数据类型[{}]".format(type(self).__name__, type(data).__name__))
        self.limit = limit
        self.data = data[:min(data.shape[0], limit)]

    def sample_data_list(self):
        return self.data.tolist()

    @classmethod
    def get_data_type(cls):
        return cls.DATA_TYPE

    def get_features(self):
        return [Field(self.DATA_TYPE, self.data.dtype.name, self.data.shape)]

    def get_input_features(self, input_cols=None):
        return self.get_features()


def _is_compatible_with_element_of_dict_or_list(element_value):
    """
    当模型输入X为dict或list类型时，使用这个方法检查是否支持里面的值

    校验规则：
       - pandas DataFrame：满足行列都大于0
       - pyspark DataFrame：满足行列都大于0
       - numpy ndarray：满足维度大于0，且第0维度len大于0
       - 单值类型：python数值型、numpy数值型、python字符串类型
       - 其它类型都算不支持的

    Args:
        element_value: 需要校验的值

    Returns: True：支持； False：不支持

    """

    from dc_model_repo.base.mr_log import logger
    # pandas DataFrame：满足行列都大于0，采样取第一行
    if isinstance(element_value, pd.DataFrame):
        ok = element_value.shape[0] > 0 and element_value.shape[1] > 0
        if not ok:
            logger.info("当前值为pd.DataFrame类型，没有满足行列都大于0的条件！")
        return ok

    # pyspark DataFrame：满足行列都大于0，采样取第一行
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    if isinstance(element_value, SparkDataFrame):
        ok = element_value.count() > 0 and len(element_value.columns) > 0
        if not ok:
            logger.info("当前值为spark DataFrame类型，没有满足行列都大于0的条件！")
        return ok

    # numpy ndarray：满足维度大于0，且第0维度len大于0。采样时候取第0个位置
    if isinstance(element_value, np.ndarray):
        ok = element_value.ndim > 0 and element_value.shape[0] > 0
        if not ok:
            logger.info("当前值为numpy ndarray类型，没有满足行列都大于0的条件！")
        return ok

    # 单值类型：python数值型、numpy数值型、python字符串类型。采样取该值
    import numbers
    if isinstance(element_value, numbers.Real):
        return True
    if isinstance(element_value, (np.int, np.float)):
        return True
    if isinstance(element_value, six.string_types):
        return True
    import datetime
    if isinstance(element_value, (datetime.datetime, datetime.date, datetime.time)):
        return True
    if isinstance(element_value, np.datetime64):
        return True

    return False


def _sample_from_compatible_element_of_dict_or_list(element_value, limit):
    """
    当模型输入X为dict或list类型时，使用这个方法对里面的元素值进行采样

    取样规则：
       - pandas DataFrame：满足行列都大于0，采样取第一行
       - pyspark DataFrame：满足行列都大于0，采样取第一行
       - numpy ndarray：满足维度大于0，且第0维度len大于0。采样时候取第一个位置
       - 单值类型：python数值型、numpy数值型、python字符串类型。采样取该值

    如果有设置了limit，取样个数为min(limit，总样本数)

    Args:
        element_value: 要采样的元素值
        limit: 采样个数

    Returns: tuple (struct, value_type, origin_shape, sample_origin, sample_plain)
    """
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    from dc_model_repo.base import Field

    struct = None
    value_type = None
    origin_shape = None
    if isinstance(element_value, pd.DataFrame):
        # 取样
        sample_count = min(element_value.shape[0], limit)
        sample_origin = element_value.iloc[:sample_count, :]
        # 转list
        sample_plain = [row.to_dict() for _, row in sample_origin.iterrows()]
        origin_shape = element_value.shape
        raise Exception("Not support pd.DataFrame")
    elif isinstance(element_value, SparkDataFrame):
        # 取样，使用limit比take快，但是不能保证是前几个数据
        sample_origin = element_value.limit(min(element_value.count(), limit)).collect()
        # 转list
        sample_plain = [row.asDict(recursive=False) for row in sample_origin]
        origin_shape = (element_value.count(), len(element_value.columns))
        raise Exception("Not support SparkDataFrame")
    elif isinstance(element_value, np.ndarray) and element_value.ndim > 0:
        sample_origin = element_value[:min(element_value.shape[0], limit)]
        sample_plain = sample_origin.tolist()
        origin_shape = element_value.shape
        struct = Field.STRUCT_NDARRAY
        value_type = element_value.dtype.name
    else:
        struct = Field.STRUCT_VAR
        value_type = type(element_value).__name__
        sample_origin = element_value
        sample_plain = sample_origin

    return struct, value_type, origin_shape, sample_origin, sample_plain


class DictDataSampler(DataSampler):
    DATA_TYPE = DatasetType.Dict

    @staticmethod
    def is_compatible(data):
        from dc_model_repo.base.mr_log import logger
        if not isinstance(data, dict):
            return False
        for k, v in data.items():
            # key必须是字符串
            if not str_util.check_is_string(k):
                logger.info("字典的键类型不兼容。应该为字符串类型，却为：{}".format(str(type(k))))
                return False
            # 校验value的类型
            if not _is_compatible_with_element_of_dict_or_list(v):
                logger.info("当前键[{}]对应的值类型[{}]不符合Dict输入要求!".format(k, str(type(v))))
                return False
        return True

    def __init__(self, data, limit=1):
        super(DictDataSampler, self).__init__(data, limit)
        if not self.is_compatible(data):
            raise Exception("{}不兼容当前数据类型[{}]\n".format(type(self), type(data)))
        self.limit = limit
        self.data = {}  # 存取样结果
        self._data_list = {}  # 存对取样结果里value可以转为list的数据进行转换后的结果
        self._dtypes = {}  # 存各value的数据类型
        for k, v in data.items():
            struct, value_type, origin_shape, sample_origin, sample_plain = _sample_from_compatible_element_of_dict_or_list(v, limit=limit)
            self._dtypes[k] = (struct, value_type, origin_shape)
            self.data[k] = sample_origin
            self._data_list[k] = sample_plain

    @classmethod
    def get_data_type(cls):
        return cls.DATA_TYPE

    def get_features(self):
        fields = []
        for k, v in self._dtypes.items():
            fields.append(Field(name=k, type=v[1], shape=v[2], struct=v[0]))
        return fields

    def sample_data_list(self):
        return self._data_list

    def to_json_str(self):
        return json_util.to_json_str(self._data_list)


class ListDataSampler(DataSampler):
    DATA_TYPE = DatasetType.List

    @staticmethod
    def is_compatible(data):
        if not isinstance(data, list):
            return False
        for v in data:
            if not _is_compatible_with_element_of_dict_or_list(v):
                return False
        return True

    def __init__(self, data, limit=1):
        super(ListDataSampler, self).__init__(data, limit)
        if not self.is_compatible(data):
            raise Exception("{}不兼容当前数据类型[{}]\n".format(type(self).__name__, type(data).__name__))
        self.limit = limit
        self.data = {}  # 存取样结果
        self._data_list = {}  # 存对取样结果里value可以转为list的数据进行转换后的结果
        self._dtypes = {}  # 存各value的数据类型
        for k, v in enumerate(data):
            struct, value_type, origin_shape, sample_origin, sample_plain = _sample_from_compatible_element_of_dict_or_list(v, limit=limit)
            self._dtypes[k] = (struct, value_type, origin_shape)
            self.data[k] = sample_origin
            self._data_list[k] = sample_plain

    @classmethod
    def get_data_type(cls):
        return cls.DATA_TYPE

    def get_features(self):
        fields = []
        for k, v in self._dtypes.items():
            fields.append(Field(name=k, type=v[1], shape=v[2], struct=v[0]))
        return fields

    def get_input_features(self, input_cols=None):
        return self.get_features()

    def sample_data_list(self):
        return self._data_list

    def to_json_str(self):
        return json_util.to_json_str(self._data_list)


def get_appropriate_sampler(X, limit=1):
    for sampler in [PandasDataFrameDataSampler, PySparkDataFrameDataSampler, NumpyDataSampler, DictDataSampler, DaskDataFrameDataSampler, ListDataSampler]:
        if sampler.is_compatible(X):
            return sampler(X, limit=limit)
    raise Exception("针对当前数据[{}.{}]找不到合适的sampler，请传入支持的数据！".format(type(X).__module__, type(X).__name__))


if __name__ == "__main__":
    samplers = []
    np_a = np.arange(10).reshape((2, 5))
    np_sample1 = NumpyDataSampler(np_a)
    np_sample2 = NumpyDataSampler(np_a, 2)
    samplers.append(np_sample1)
    samplers.append(np_sample2)

    pd_df = pd.DataFrame(np.eye(3), columns=["a", "b", "c"])
    import pyspark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(pd_df)
    dict_input = {"pandas": pd_df,
                  "spark": spark_df,
                  "numpy": np.ones((3, 5)),
                  "numpy1": np.arange(10),
                  "int": 6, "float": 6.6,
                  "np_int": np.int_(9), "np_float": np.float_(9.9),
                  "string": "hello"}
    dict_sample1 = DictDataSampler(dict_input)
    dict_sample2 = DictDataSampler(dict_input, 2)
    samplers.append(dict_sample1)
    samplers.append(dict_sample2)

    # spark_df1 = spark.createDataFrame(pd_df)
    spark_df1 = spark_df
    list_input = [pd_df, spark_df1, np.ones((3, 5)), np.array([0]), 5, 6.6, np.int_(9), np.float_(9.9), "hello"]
    list_sample1 = ListDataSampler(list_input)
    list_sample2 = ListDataSampler(list_input, limit=3)
    samplers.append(list_sample1)
    samplers.append(list_sample2)

    for s in samplers:
        print("\n\n\n{:-^55} limit={}".format(type(s).__name__, s.limit))
        print("origin:")
        print(s.data)
        print("data converted to list:")
        print(s.sample_data_list())
        print("features:")
        for f in s.get_features():
            print(str(f))
        print("json string:")
        print(s.to_json_str())
