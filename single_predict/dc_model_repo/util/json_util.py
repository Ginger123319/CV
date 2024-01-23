# -*- encoding: utf-8 -*-

import json
from json import JSONEncoder
from dc_model_repo.util import str_util


class RTEncoder(JSONEncoder):
    def default(self, o):
        _cls = o.__class__
        _class_name = _cls.__module__ + "." + _cls.__name__
        if _class_name == 'pandas._libs.tslibs.timestamps.Timestamp' or _class_name == 'pandas.tslib.Timestamp':
            return str(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)


def to_json_str(d):
    """
    防止生成Unicode
    :param d:
    :return:
    """
    import six
    if six.PY2:
        return json.dumps(d, ensure_ascii=False, cls=RTEncoder)
    else:
        return json.dumps(d, ensure_ascii=False, cls=RTEncoder)


def to_json_bytes(d):
    """
    防止生成Unicode
    :param d:
    :return:
    """
    str_data = to_json_str(d)
    from dc_model_repo.util import str_util
    return str_util.to_bytes(str_data)


def to_object(s):
    """
    防止生成Unicode
    :param s:
    :return:
    """
    d = json.loads(s)
    return str_util.byteify(d)
