# -*- encoding: utf-8 -*-
import numpy as np


def to_dataframe(data, name):
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        print(hasattr(data, "shape"))
        if hasattr(data, "shape"):
            if data.shape.__len__() == 2:
                column_size = data.shape[1]
            else:
                column_size = 1
        else:
            column_size = len(data)
        columns = []
        for i in range(column_size):
            columns.append("%s%s" % (name, i))
        data = pd.DataFrame(data, columns=columns)
    else:
        columns = data.columns.values
    return data, columns


# 多维数据合并
def array_concat(data_x, data_y):
    if data_x is None and data_y is None:
        return []

    if data_x is None:
        return data_y
    if data_y is None:
        return data_x

    limit = 5
    if not isinstance(data_x, np.ndarray):
        data_x = np.linspace(data_x, data_x, num=limit)
    if not isinstance(data_y, np.ndarray):
        data_y = np.linspace(data_y, data_y, num=limit)

    data_x = data_x[:limit]
    data_y = data_y[:limit]

    limit = len(data_x)
    if len(data_y) < limit:
        limit = len(data_y)
        data_x = data_x[:limit]
        data_y = data_y[:limit]

    new_list = []
    for i in range(limit):
        new_list.append([data_x[i], data_y[i]])

    return new_list
