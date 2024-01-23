# -*- encoding: utf-8 -*-
from dc_model_repo.step.base import BaseTransformer, BaseEstimator

from dc_model_repo.base import StepType, FrameworkType, Field, Output, ChartData, ModelFileFormatType

import pandas as pd
import abc, six


@six.add_metaclass(abc.ABCMeta)
class PandasDCStep(BaseTransformer):
    """抽象基于Pandas DataFrame的Step 公共特性。
    """

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

