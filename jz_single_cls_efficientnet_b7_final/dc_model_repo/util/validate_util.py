# -*- encoding: utf-8 -*-


def require_type(name, o, t):
    if o is not None:
        if not isinstance(o, t):
            raise Exception("'%s'需要%s类型。" % (name, t.__name__))


def require_attr_not_none(o, name):
    """校验对象中的属性不能为空。
    Args:
        o:
        name: 属性的名称。
    Returns:
    """
    if o is not None:
        if getattr(o, name, None) is None:
            raise Exception("对象=%s的属性'%s'不能为空。" % (str(o), name))


def require_list_non_empty(name, o):
    """校验数组不能为空。
    Args:
        name: 提示对象名称。
        o: 数组对象。
    Returns:
    """
    if is_non_empty_list(o):
        pass
    else:
        raise Exception("'%s' 不能为空。" % name)


def require_str_non_empty(str_obj, tips):
    """校验数组不能为空。
    Args:
        str_obj: 字符串对象。
        tips: 为空时的提示信息。
    Returns:
    """
    if str_obj is None or len(str_obj) == 0:
        raise Exception("'%s' 不能为空。" % tips)


def is_non_empty_list(o):
    return o is not None and len(o) > 0


def is_empty_list(o):
    return o is None or len(o) == 0


def is_non_empty_str(o):
    return o is not None and isinstance(o, str) and len(o) > 0

