import abc
import os
from os import path as P
import six

from .base import BaseTransformer, BaseEstimator
from ..base import FrameworkType, ModelFileFormatType, LearningType
from ..util import cls_util
from ..base.mr_log import logger
from ..util.decorators import keyword_only

CONST_IGNORE_SOURCE_CODE_PATH = "ignore"


def _fix_source_code_path(origin_path, obj):
    """修正源码路径

    检查给出的源码路径是否存在，如果没有为None或者不存在，会根据当前类的根包对应的路径当作源码路径。
    之后再将路径转换为标准绝对路径，且保证路径末尾没有"/"或"\\"

    Args:
        origin_path: 原始路径。

    Returns: 经过处理的路径
    """

    if origin_path is None:
        origin_path = cls_util.get_source_module(obj)
        class_name = cls_util.get_full_class_name(obj)
        msg = "这是个文件" if os.path.isfile(origin_path) else "这是个目录"
        logger.info("检测到自定义Transformer类[%s]的源码路径为[%s] [%s]" % (class_name, origin_path, msg))
    elif not os.path.exists(origin_path):
        logger.info("设置的源码路径[%s]不存在，将自动检测源码路径" % origin_path)
        origin_path = None

    # 取绝对路径
    source_code_path_abs = os.path.abspath(origin_path)
    # 去掉末尾的("/", "\\") 防止P.basename("dir/") 得到的值为 ""
    while len(source_code_path_abs) > 1 and any([source_code_path_abs.endswith(x) for x in ("/", "\\")]):
        source_code_path_abs = source_code_path_abs[:-1]
    if origin_path != source_code_path_abs:
        logger.info("源码路径[%s]已被转换成了标准绝对路径[%s]" % (origin_path, source_code_path_abs))
    return source_code_path_abs


@six.add_metaclass(abc.ABCMeta)
class UserDefinedTransformer(BaseTransformer):
    """用户自定义Transformer的基类。

    用户需要复写的情况：

        - 需要自己初始化一些额外信息。
        - 且需要调用当前方法。

    Args:
        input_cols: 输入列的名称列表
        algorithm_name: 算法名称，为None时取实现类的类名作为默认值
        model_format: 模型类型，可选值见 :class:`dc_model_repo.base.ModelFileFormatType` 中的取值。

            - 如果为None，会取默认值ModelFileFormatType.USER_DEFINED。
            - 如果模型格式为“h5",设置此值为ModelFileFormatType.H5，以支持模型可视化。
            - 如果模型格式为“PB",设置此值为ModelFileFormatType.PB，以支持模型可视化。

        source_code_path: 自定义Transformer的源码路径，可以是文件或者目录。在持久化过程中会将该路径下的文件拷贝到模型文件中。

            - 如果为None，会取自定义Transformer的根包路径作为默认值。
            - 如果为 :attr:`IGNORE_SOURCE_CODE_PATH` ：不复制source code

        extension: 扩展信息
        requirements: 该Transformer依赖的python包。
        output_type: 如果输出的数据类型与输入的不一致，这里才需要设置。取值见：:class:`dc_model_repo.base.DatasetType`
        framework: 当前step使用什么框架，默认值为 :attr:`dc_model_repo.base.FrameworkType.Custom` 。目前建议不要配置。
        **kwargs: 预留参数位置
    """

    IGNORE_SOURCE_CODE_PATH = CONST_IGNORE_SOURCE_CODE_PATH

    @keyword_only(ignore_self=True)
    def __init__(self,
                 input_cols=None,
                 algorithm_name=None,
                 model_format=None,
                 source_code_path=None,
                 extension=None,
                 requirements=None,
                 output_type=None,
                 framework=None,
                 **kwargs):

        if algorithm_name is None:
            algorithm_name = self.__class__.__name__

        if model_format is None:
            model_format = ModelFileFormatType.USER_DEFINED

        if source_code_path != UserDefinedTransformer.IGNORE_SOURCE_CODE_PATH:
            source_code_path = _fix_source_code_path(source_code_path, self)
        else:
            source_code_path = None
            logger.info("当前自定义transformer设置了忽略源码路径: {}".format(type(self)))

        if framework is None:
            framework = FrameworkType.Custom

        super(UserDefinedTransformer, self).__init__(operator=None,
                                                     framework=framework,
                                                     model_format=model_format,
                                                     input_cols=input_cols,
                                                     algorithm_name=algorithm_name,
                                                     source_code_path=source_code_path,
                                                     extension=extension,
                                                     requirements=requirements,
                                                     output_type=output_type,
                                                     **kwargs)

    def fit(self, X, y=None, options=None, **kwargs):
        """训练模型。

        用户需要复写的情况：

        - 算子执行transform时需要从训练数据中获取额外的信息时要复写。
        - 且要先调用super(YourTransformer, self).fit(X=X,y=y)之后再写其它逻辑。
        - 一般来说，这时这些信息需要挂载到self上，并配合persist_model一起使用，以便在persist_model方法执行时进行持久化。

        Args:
            X : 特征数据
            y (ndarray): 标签数据。
            options(dict): 送到模型中的参数。
            **kwargs: 备用扩展

        Returns: self。
        """
        super(UserDefinedTransformer, self).fit(X=X, y=y, options=options, **kwargs)
        return self

    @abc.abstractmethod
    def transform(self, X, **kwargs):
        """转换数据。

        用户需要复写。

        Args:
            X: 需要转换的数据
            kwargs: 备用扩展

        Returns: 转换后的数据
        """
        pass

    def persist_model(self, fs, destination):
        """
        持久化算子fit时产生的额外信息。

        用户需要复写的情况：

        - 如果当前算子在fit时候会产生一些在之后转换数据时需要的额外信息，在这里将其持久化。

        Args:
            fs: 当前运行时存放模型的文件管理器代理，为 :class:`dc_model_repo.base.file_system.DCFileSystem` 的实现类。
                通常:

                - 分布式环境为 :class:`dc_model_repo.base.file_system.HDFSFileSystem`，
                - 单机环境为 :class:`dc_model_repo.base.file_system.LocalFileSystem`

            destination:当前运行时保存模型文件的路径
        """
        pass

    def prepare(self, step_path, **kwargs):
        """加载persist_model阶段保存的模型

        用户需要复写的情况：

        - 在fit时产生了一些额外的信息，并在persist_model方法中对其进行了持久化。

        Args:
            step_path: 模型路径
            **kwargs: 备用扩展
        """
        pass

    def get_persist_step_ignore_variables(self):
        """
        设置保存当前算子的step时需要忽略的属性。

        用户一般不需要复写该方法，详细介绍如下：

        在持久化模型时候，每个算子是分成两部分进行持久化的，一部分是当前step的APS框架信息，一部分是当前step所挂载的模型信息。

        通过这个方法可以控制哪些属性不被持久化到APS框架信息中，而这些你不想持久化的属性又可以大致分为两类：

            1. 这个属性是你自定义的模型文件，你想在self.persist_model方法中把它单独保存到一个地方。
            2. 这个属性由于某种原因没法使用python的pickle机制进行持久化，这时必须把这个属性设置到ignore列表中，不然程序会报错。

        此外，如果模型正常运转需要这里忽略的属性，你需要：

            1. 在self.persist_model中单独持久化这些属性；
            2. 在self.prepare中加载这些属性内容，并挂载到self中


        Returns:
            list: list of str
        """
        pass

    def get_params(self):
        """提供当前模型的参数信息，这些信息会被记录到模型的meta.json中。

        用户一般不用复写。

        Returns:
            list: list of :class:`dc_model_repo.base.Param`
        """
        pass


@six.add_metaclass(abc.ABCMeta)
class UserDefinedEstimator(BaseEstimator):
    """用户自定义Estimator的基类。

    用户需要复写的情况：

    - 需要自己初始化一些额外信息。
    - 且需要调用当前方法。

    Args:
        input_cols: 输入列的名称列表
        target_cols: 标签列的名称列表
        output_cols: 输出列的名称列表
        learning_type: 当前算法的学习类型。可选值见 :class:`dc_model_repo.base.LearningType` ，如果为None，会设置成默认值Unknown
        algorithm_name: 算法名称，为None时取实现类的类名作为默认值
        model_format: 模型类型，可选值见 :class:`dc_model_repo.base.ModelFileFormatType`

            - 如果为None，会取默认值 :attr:`dc_model_repo.base.ModelFileFormatType.USER_DEFINED`
            - 如果模型格式为“h5",设置此值为ModelFileFormatType.H5，以支持模型可视化。
            - 如果模型格式为“PB",设置此值为ModelFileFormatType.PB，以支持模型可视化。

        source_code_path: 自定义Transformer的源码路径，可以是文件或者目录。在持久化过程中会将该路径下的文件拷贝到模型文件中。

            - 如果为None，会取自定义Transformer的根包路径作为默认值。
            - 如果为 :attr:`IGNORE_SOURCE_CODE_PATH` ，不复制source code

        extension: 扩展信息
        requirements: 该Transformer依赖的python包
        output_type: 取值见：:class:`dc_model_repo.base.DatasetType`

          - 必须是DatasetType.PandasDataFrame, DatasetType.PySparkDataFrame, DatasetType.Dict中的一个。
          - 如果输出的数据类型与输入的一致且满足上述要求，这里可以不设置。

        framework: 当前Estimator使用什么框架，默认值为 :attr:`dc_model_repo.base.FrameworkType.Custom` 。目前建议不要配置。
        **kwargs: 预留参数位置
    """

    IGNORE_SOURCE_CODE_PATH = CONST_IGNORE_SOURCE_CODE_PATH

    @keyword_only(ignore_self=True)
    def __init__(self,
                 input_cols=None,
                 output_cols=None,
                 target_cols=None,
                 learning_type=None,
                 algorithm_name=None,
                 model_format=None,
                 source_code_path=None,
                 extension=None,
                 requirements=None,
                 output_type=None,
                 framework=None,
                 **kwargs):

        if learning_type is None:
            learning_type = LearningType.Unknown

        if algorithm_name is None:
            algorithm_name = self.__class__.__name__

        if model_format is None:
            model_format = ModelFileFormatType.USER_DEFINED

        if source_code_path != UserDefinedEstimator.IGNORE_SOURCE_CODE_PATH:
            source_code_path = _fix_source_code_path(source_code_path, self)
        else:
            source_code_path = None
            logger.info("当前自定义estimator设置了忽略源码路径: {}".format(type(self)))

        if framework is None:
            framework = FrameworkType.Custom

        if output_type is not None:
            assert output_type in BaseEstimator.ALLOWED_OUTPUT_TYPE, "Estimator的输出类型output_type必须是{}中的一个，现在却为：{}。可在初始化Step时使用参数output_type明确指定!".format(repr(BaseEstimator.ALLOWED_OUTPUT_TYPE), repr(output_type))

        super(UserDefinedEstimator, self).__init__(operator=None,
                                                   input_cols=input_cols,
                                                   output_cols=output_cols,
                                                   target_cols=target_cols,
                                                   learning_type=learning_type,
                                                   framework=framework,
                                                   model_format=model_format,
                                                   algorithm_name=algorithm_name,
                                                   source_code_path=source_code_path,
                                                   extension=extension,
                                                   requirements=requirements,
                                                   output_type=output_type,
                                                   **kwargs)

    def fit(self, X, y=None, options=None, **kwargs):
        """训练模型。

        用户需要复写。

        Args:
            X : 特征数据
            y (ndarray): 标签数据。
            options(dict): 送到模型中的参数。
            **kwargs: 备用扩展

        Returns: self。
        """
        super(UserDefinedEstimator, self).fit(X=X, y=y, options=options, **kwargs)
        return self

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        """进行预测。

        用户需要复写。

        Args:
            X: 需要预测的数据
            kwargs: 备用扩展

        Returns: 预测结果
        """
        pass

    def persist_model(self, fs, destination):
        """持久化算子fit时产生的额外信息。

        用户需要复写的情况：

        - 如果当前算子在fit时候会产生一些在之后转换数据时需要的额外信息，在这里将其持久化。

        Args:
            fs: 当前运行时存放模型的文件管理器代理，为 :class:`dc_model_repo.base.file_system.DCFileSystem` 的实现类。
                通常:

                - 分布式环境为 :class:`dc_model_repo.base.file_system.HDFSFileSystem`，
                - 单机环境为 :class:`dc_model_repo.base.file_system.LocalFileSystem`

            destination:当前运行时保存模型文件的路径
        """
        pass

    def prepare(self, step_path, **kwargs):
        """加载persist_model阶段保存的模型

        用户需要复写的情况：

        - 在fit时产生了一些额外的信息，并在persist_model方法中对其进行了持久化。

        Args:
            step_path: 模型路径
            **kwargs: 备用扩展
        """
        pass

    def get_persist_step_ignore_variables(self):
        """设置保存当前算子的step时需要忽略的属性。

        用户一般不需要复写该方法，详细介绍如下：

        在持久化模型时候，每个算子是分成两部分进行持久化的，一部分是当前step的APS框架信息，一部分是当前step所挂载的模型信息。
        通过这个方法可以控制哪些属性不被持久化到APS框架信息中，而这些你不想持久化的属性又可以大致分为两类：

        1. 这个属性是你自定义的模型文件，你想在self.persist_model方法中把它单独保存到一个地方。
        2. 这个属性由于某种原因没法使用python的pickle机制进行持久化，这时必须把这个属性设置到ignore列表中，不然程序会报错。

        此外，如果模型正常运转需要这里忽略的属性，你需要：

        1. 在self.persist_model中单独持久化这些属性；
        2. 在self.prepare中加载这些属性内容，并挂载到self中

        Returns:
            list: list of str
        """
        pass

    def get_params(self):
        """
        提供当前模型的参数信息，这些信息会被记录到模型的meta.json中。

        用户一般不用复写。

        Returns: 需要被记录的参数数，数据类型为list，其中的元素的类型需为`dc_model_repo.base.Param`
        """
        pass
