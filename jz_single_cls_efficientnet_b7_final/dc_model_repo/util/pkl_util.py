# -*- encoding: utf-8 -*-
import six
import pickle
ENCODING_LIST = ["iso-8859-1", "ascii", 'utf-8', "gbk", "gb2312", "gb18030"]

PICKLE_PROTOCOL = 2
if six.PY2:
    PICKLE_PROTOCOL = 2
elif six.PY3:
    PICKLE_PROTOCOL = 3


def serialize_with_ignore_variables(obj, variables):
    """
    序列化对象时忽略部分属性。
    :param obj:
    :param variables:
    :return:
    """
    if variables is None:
        variables = []
    cache_map = {}
    # 1. 忽略对象
    for v_name in variables:
        if hasattr(obj, v_name):
            value = getattr(obj, v_name)
            cache_map[v_name] = value
            setattr(obj, v_name, None)
    # 2. 导出数据
    bytes_value = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)

    # 3. 还原对象
    for k in cache_map:
        setattr(obj, k, cache_map[k])

    return bytes_value


def deserialize(data):
    if six.PY2:
        return pickle.loads(data)
    else:
        _e = None
        for encoding in ENCODING_LIST:
            try:
                obj = pickle.loads(data, encoding=encoding)
                return obj
            except Exception as e:
                _e = e
                print("使用编码%s加载对象失败， 原因 %s。" % (encoding, str(e)))
        raise _e


def deserialize_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        return deserialize(data)


def serialize2bytes(obj):
    return serialize_with_ignore_variables(obj, None)


def serialize2file(obj, path):
    data = serialize_with_ignore_variables(obj, None)
    with open(path, 'wb') as f:
        f.write(data)
