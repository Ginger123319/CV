# -*- encoding: utf-8 -*-

ENV_EXTENSION_COLLECTOR = "MR_SDK_EXTENSION_COLLECTOR"
ENV_STEP_COLLECTOR = "MR_SDK_STEP_COLLECTOR"


# 单例插件
def load_collect(env_key):
    import os
    # 要求类已经在PythonPath中， 配置的为类名
    from dc_model_repo.base.mr_log import logger
    collector_name = os.environ.get(env_key)
    # 不能使用cls_util, 否则导致依赖冲突 from dc_model_repo.util import cls_util
    if collector_name and len(collector_name) > 0:
        from dc_model_repo.util import cls_util
        collector = cls_util.get_class(collector_name)()
        if env_key == ENV_EXTENSION_COLLECTOR:
            from dc_model_repo.base.collector import ExtensionCollector
            if isinstance(collector, ExtensionCollector):
                logger.info("读取扩展信息采集插件:%s=%s。" % (env_key, collector_name))
                return collector
            else:
                raise Exception('Class [%s] is not extends "ExtensionCollector" .')
        elif env_key == ENV_STEP_COLLECTOR:
            from dc_model_repo.base.collector import StepCollector
            if isinstance(collector, StepCollector):
                logger.info("读取到Step管理插件:%s=%s." % (env_key, collector_name))
                return collector
            else:
                raise Exception('Class [%s] is not extends "StepCollector" .')
        else:
            raise Exception('不支持环境变量： %s' % env_key)
    else:
        # logger.warning("检测到没有配置环境变量: %s." % env_key)
        return None


extension_collector = load_collect(ENV_EXTENSION_COLLECTOR)
step_collector = load_collect(ENV_STEP_COLLECTOR)


def get_extension_collector():
    global extension_collector
    if extension_collector is None:
        extension_collector = load_collect(ENV_EXTENSION_COLLECTOR)
    return extension_collector


def get_step_collector():
    global step_collector
    if step_collector is None:
        step_collector = load_collect(ENV_STEP_COLLECTOR)
    return step_collector

