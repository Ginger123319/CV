# -*- encoding: utf-8 -*-

import abc


class ExtensionCollector(object):
    """
    搜集Transformer的环境信息
    """

    @abc.abstractmethod
    def collect(self):
        pass

    @abc.abstractmethod
    def get_running_environment(self):
        """定义插件的运行环境名称。
        Returns:
        参考dc_model_repo.base.RunningEnvironmentType
        """
        pass

    @abc.abstractmethod
    def get_steps_path(self):
        pass

    @abc.abstractmethod
    def get_data_path(self):
        pass

    @abc.abstractmethod
    def get_module_id(self):
        pass

    @abc.abstractmethod
    def get_mr_server_portal(self):
        pass

    @abc.abstractmethod
    def get_model_repo_name(self):
        pass


class StepCollector(object):
    """
    多进程环境中收集落地的Transformers
    """
    @abc.abstractmethod
    def collect_pipeline(self):
        pass

