import functools
import warnings

warnings.filterwarnings("always", message="This (class|method) .* is deprecated")


def deprecated(_func=None, msg=""):
    def real_decorator(func):
        import inspect
        if inspect.isclass(func):
            warnings.warn("This class [{}] is deprecated. {}".format(func.__qualname__, msg), DeprecationWarning, stacklevel=3)
            return func
        else:
            def wrapper(*args, **kwargs):
                if func.__name__ == "__init__":
                    warnings.warn("This class [{}] is deprecated. {}".format(func.__qualname__[:-9], msg), DeprecationWarning, stacklevel=2)
                else:
                    warnings.warn("This method [{}] is deprecated. {}".format(func.__qualname__, msg), DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return wrapper

    if _func is None:  # 这时是@deprecated()
        return real_decorator
    else:  # 这时是@deprecated
        return real_decorator(_func)


def keyword_only(ignore_self=False):
    def real_keyword_only(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if (ignore_self and len(args) > 1) or (not ignore_self and len(args) > 0):
                raise ValueError("方法[{}]必须使用关键字传参".format(func.__qualname__))
            return func(*args, **kwargs)

        return wrapper

    if isinstance(ignore_self, bool):
        return real_keyword_only
    else:
        return real_keyword_only(ignore_self)


if __name__ == "__main__":
    from dc_model_repo.base.data_sampler import ArrayDataSampler

    print("temp")
    temp = ArrayDataSampler([1, 2, 4])
    print("temp1")
    temp1 = ArrayDataSampler([1, 2, 3])


    @deprecated
    def world(a, b):
        pass

    world()

    @keyword_only
    def hello(a, b):
        print(a, b)


    hello(a="a", b="b")
    hello("a", "b")

    print("END")
