# -*- encoding: utf-8 -*-

from py4j.java_collections import JavaMap, JavaArray
from py4j.java_gateway import JavaObject


def __try_convent_java_array__(java_object):
    def _convent_(obj):
        if isinstance(obj, JavaArray):
            return list(obj)
        elif isinstance(obj, JavaObject):
            return str(obj)
        else:
            return obj

    if isinstance(java_object, list):
        return [_convent_(j_a) for j_a in java_object]
    else:
        return _convent_(java_object)


def __try_convent_java_map__(map):
    if isinstance(map, JavaMap):
        _d = {}
        for k, v in list(map.items()):
            _d[k] = __try_convent_java_array__(v)
        return _d
    else:
        return map
