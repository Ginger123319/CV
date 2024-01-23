# -*- encoding: utf-8 -*-
import pandas as pd


def cast_df_type(df, features):
    """对df中的列进行类型转换。
    Args:
        df:
        features:
    Returns:
    """
    from dc_model_repo.base.mr_log import logger
    logger.info("类型转换前: \n%s" % str(df.dtypes))
    for feature in features:
        name = feature.name
        type = feature.type
        try:
            if type.startswith("datetime"):
                df[name] = pd.to_datetime(df[name], errors="coerce")
            elif type.startswith("int"):
                df[name] = df[[name]].astype('float')
            else:
                df[name] = df[[name]].astype(type)
        except Exception as e:
            pass
    logger.info("类型转换后: \n%s" % str(df.dtypes))
    return df
