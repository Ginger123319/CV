# -*- encoding: utf-8 -*-
from os import path as P


def get_class_name(o):
    _cls = o.__class__
    return _cls.__name__


def get_module_name(o):
    return o.__class__.__module__


def get_full_class_name(o):
    return o.__class__.__module__ + "." + get_class_name(o)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    modules = parts[1:]
    for comp in modules:
        m = getattr(m, comp)
    return m


def get_source_module(obj):
    """从对象中找到对应的模块文件路径。

    Args:
        obj:

    Returns:

    """
    module_path = __import__(obj.__module__).__file__
    if P.basename(module_path) == "__init__.py":
        module_path = P.dirname(module_path)

    return module_path


def get_source_file(obj):
    """从对象中找到对应的class文件路径。

    Args:
        obj:

    Returns:

    """
    module_name = obj.__module__
    parts = module_name.split('.')

    m = __import__(module_name)
    modules = parts[1:]
    for comp in modules:
        m = getattr(m, comp)
    return m


def language():
    import six
    if six.PY2:
        return 'python2'
    elif six.PY3:
        return 'python3'
    else:
        return 'python'
