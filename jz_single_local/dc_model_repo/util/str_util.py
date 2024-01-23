# -*- encoding: utf-8 -*-
import six


def to_str(sv):
    """将unicode和python3中的字节转换成字符串。

    Args:
        sv(Union(bytes, unicode, object)): 字节、unicode或者其他类型的数据转换为字符串；

    Returns:
        str: 字符串数据。
    """
    if six.PY2:
        if isinstance(sv, unicode):
            return sv.encode('utf-8')
        else:
            return str(sv)
    else:  # 在py3以及更高的版本中
        if isinstance(sv, bytes):
            return str(sv, encoding='utf-8')
        else:
            return str(sv)


def to_bytes(s):
    """将字符串转换为字节数组。

    Args:
        s (Union(str, unicode)): 需要转换为字节的数据，在python2中支持类型str和unicode；在py3中支持str。

    Returns:
        字节数据。
    """
    if six.PY2:
        # 在python2中字符串就是字节数组
        if isinstance(s, unicode):
            return s.encode('utf-8')
        elif isinstance(s, str):
            return s
        else:
            raise Exception("无法将类型%s转换为字节" % type(s).__name__)
    else:  # 在py3以及更高的版本中
        if isinstance(s, str):
            return bytes(s, encoding="utf-8")
        elif isinstance(s, bytes):
            return s
        else:
            raise Exception("无法将类型%s转换为字节" % type(s).__name__)


def byteify(s, encoding='utf-8'):
    """
    把Dict中的Unicode转换为字符串
    :param s:
    :param encoding:
    :return:
    """
    if isinstance(s, dict):
        r = {}
        for k in s:
            r[byteify(k)] = byteify(s[k])
        return r
    elif isinstance(s, list):
        return [byteify(element) for element in s]
    elif type(s).__name__ == 'unicode':
        return s.encode(encoding)
    else:
        return s


def check_is_string(s):
    return isinstance(s, six.string_types)
