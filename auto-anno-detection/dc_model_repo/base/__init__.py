# -*- encoding: utf-8 -*-
import abc
import six

from dc_model_repo.util import str_util


class StepType(object):
    Transformer = 'transformer'
    Estimator = 'estimator'


class FrameworkType(object):
    SKLearn = "SKLearn"
    HyperGBM = "HyperGBM"
    DeepTables = "DeepTables"
    Spark = "Spark"
    TensorFlow = "TensorFlow"
    TensorFlow2 = "TensorFlow2"
    Keras = "Keras"
    Pytorch = "Pytorch"
    Mixed = "Mixed"
    Custom = "Custom"
    APS = "APS"
    APS32 = "APS32"
    APS31Custom = "APS31Custom"
    Dask = "Dask"


class ModelFileFormatType(object):
    PKL = "pkl"
    ZIP = "zip"
    PMML = "pmml"
    ONNX = "onnx"
    H5 = "h5"
    PB = "pb"
    CKPT = "ckpt"
    SAVED_MODEL = "saved_model"
    PTH = "pth"
    T7 = "t7"
    DIR = "dir"
    USER_DEFINED = "any"


class RunningEnvironmentType(object):
    Spark = 'spark'
    Local = 'local'


class Mertics(object):
    Accuracy = "accuracy"
    Log_Loss = "log_loss"
    F1 = "f1"
    FBeta = "fbeta"
    AUC = "roc_auc"
    Recall = "recall"
    Precision = "precision"
    MSLE = "neg_mean_squared_log_error"
    RMSE = "rmse"
    MSE = "neg_mean_squared_error"
    MAE = "neg_mean_absolute_error"
    R2 = "r2"
    MedianAE = "neg_median_absolute_error"
    EVS = "explained_variance"


Metrics = Mertics


class PerformanceType(object):
    File = 'file'
    Metrics = "metrics"
    Confusion_matrix = "confusion_matrix"
    RocCurve = "roc_curve"
    PrecisionRecallCurve = "precision_recall_curve"
    KSCurve = "ks_curve"
    GainCurve = "gain_curve"
    LiftCurve = "lift_curve"
    ConfusionMatrixCut = "confusion_matrix_cut"


class ExplanationType(object):
    FeatureImportance = "feature_importance"
    Tree = "tree"
    RegressionCoefficients = "regressionCoefficients"


class TorchDType(object):
    Float = 'float'
    Long = 'long'
    Bool = 'bool'
    Byte = 'byte'
    Char = 'char'
    Int = 'int'
    Short = 'short'


class DatasetType(object):
    PandasDataFrame = 'pandasDataFrame'
    DaskDataFrame = 'daskDataFrame'
    PySparkDataFrame = 'pySparkDataFrame'
    ArrayData = 'arrayData'
    NumpyArray = "NumpyArray"
    Dict = "Dict"
    List = "List"

    @classmethod
    def all_values(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("__") and isinstance(v, str)]


@six.add_metaclass(abc.ABCMeta)
class DictSerializable(object):

    @classmethod
    def field_mapping(cls):
        raise NotImplementedError

    def to_dict(self):
        m = self.field_mapping()
        return self.member2dict(m)

    @staticmethod
    def len_construction_args():
        pass

    @classmethod
    def load_from_dict(cls, dict_data):
        if dict_data is None:
            return None
        else:
            len_construction_args = cls.len_construction_args()
            if len_construction_args is None:
                raise Exception("请实现len_construction_args()方法。")
            args = [None for i in range(len_construction_args)]
            instance = cls(*args)
            instance.dict2member(cls.field_mapping(), dict_data)
            return instance

    @classmethod
    def load_from_dict_list(cls, dict_data_list):
        return None if dict_data_list is None else [cls.load_from_dict(d) for d in dict_data_list]

    @classmethod
    def load_from_json_str(cls, json_str):
        from dc_model_repo.util import json_util
        dict_data = json_util.to_object(json_str)
        return cls.load_from_dict(dict_data)

    def member2dict(self, field_mapping):
        """当成员变量的值是基本类型时候可以使用此方法把成员变量转换成字典。

        Args:
            field_mapping:

        Returns: dict.
        """
        result = {}
        for key in field_mapping:
            result[field_mapping[key]] = getattr(self, key)
        return result

    def dict2member(self, mapping, dict_data):
        for k in mapping:
            setattr(self, k, dict_data.get(mapping[k]))

    def to_json_string(self):
        from dc_model_repo.util import json_util
        return json_util.to_json_str(self.to_dict())

    @staticmethod
    def to_dict_if_not_none(ds):
        if ds is None:
            return None
        else:
            if isinstance(ds, DictSerializable):
                return ds.to_dict()
            elif isinstance(ds, list):
                return [d.to_dict() for d in ds]
            else:
                raise Exception("不支持to_dict操作，对象是: %s" % str(ds))


class BaseOperator(object):
    """基本操作算子。
        Args:
            id (str): 对象唯一标志。
            extension (dict): 扩展属性。
    """

    FILE_DATA = "data"
    FILE_SAMPLE_DATA = "sampleData.json"
    FILE_TARGET_SAMPLE_DATA = "targetSampleData.json"
    FILE_META = "meta.json"

    RELATIVE_PATH_SAMPLE_DATA = FILE_DATA + "/" + FILE_TARGET_SAMPLE_DATA
    RELATIVE_PATH_TARGET_SAMPLE_DATA = FILE_DATA + "/" + FILE_TARGET_SAMPLE_DATA

    def __init__(self, id, extension):

        self.id = id
        self.module_id = None
        self.target_sample_data = None
        self.sample_data = None

        # 加载 env
        from dc_model_repo.base import collector_manager
        extension_collector = collector_manager.get_extension_collector()

        self.extension = extension
        envs = None
        if extension_collector is not None:
            envs = extension_collector.collect()
            self.module_id = extension_collector.get_module_id()
        if envs is None:
            envs = {}
        if self.extension is None:
            self.extension = {}
        # 合并到extension中
        for k in envs:
            self.extension[k] = envs[k]

    @staticmethod
    def join_path(*paths):
        if len(paths) > 1:
            return "/".join(paths)
        elif len(paths) == 1:
            return paths[0]
        else:
            return None

    @staticmethod
    def serialize_meta_path(destination):
        return BaseOperator.join_path(destination, BaseOperator.FILE_META)

    @staticmethod
    def serialize_data_path(destination):
        return BaseOperator.join_path(destination, BaseOperator.FILE_DATA)

    def serialize_sample_data_path(self, destination):
        return BaseOperator.join_path(self.serialize_data_path(destination), BaseOperator.FILE_SAMPLE_DATA)

    def serialize_target_sample_data_path(self, destination):
        return BaseOperator.join_path(self.serialize_data_path(destination), BaseOperator.FILE_TARGET_SAMPLE_DATA)

    def _persist_sample_data(self, fs, sample_data, sample_data_path):
        from dc_model_repo.base.mr_log import logger
        #  落地样本文件
        if sample_data is not None:
            try:
                fs.write_bytes(sample_data_path, str_util.to_bytes(sample_data.to_json_str()))
            except Exception as e:
                logger.warning("序列化样本数据失败，原因: %s" % str(e))

    def persist_sample_data(self, fs, destination, persist_sample_data):
        from dc_model_repo.base.mr_log import logger
        if persist_sample_data:
            logger.info("开始序列化样本数据。")
            self._persist_sample_data(fs, self.sample_data, self.serialize_sample_data_path(destination))
            self._persist_sample_data(fs, self.target_sample_data, self.serialize_target_sample_data_path(destination))
        else:
            logger.info("已经设置跳过序列化样本数据。")

    @staticmethod
    def get_fs_type():
        """推断当前算子应该使用的文件系统类型。
        Returns:  文件系统类型
        """
        from dc_model_repo.base import collector_manager
        extension_collector = collector_manager.get_extension_collector()
        from dc_model_repo.base import file_system
        from dc_model_repo.base.mr_log import logger
        if extension_collector is not None:
            running_environment = extension_collector.get_running_environment()
            if running_environment == RunningEnvironmentType.Spark:
                return file_system.FS_HDFS
            elif running_environment == RunningEnvironmentType.Local:
                return file_system.FS_LOCAL

        logger.warning("无法从环境信息插件推测出文件系统的类型，将使用本地文件系统。")
        return file_system.FS_LOCAL


class Param(DictSerializable):

    @classmethod
    def field_mapping(cls):
        return {
            "name": "name",
            "type": "type",
            "value": "value"
        }

    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    @classmethod
    def len_construction_args(cls):
        return 3


class Field(DictSerializable):
    """用于描述单个特征列

    Args:
        name: 列名
        type: 类型
        shape: 形状
        struct: 支持ndarray（对应numpy.ndarray)、list(对应python的list）、dict（对应python的dict）、var（对应其他类型，本意是单值类型）。
            目前又做了限定，只支持ndarray和var
    """
    STRUCT_NDARRAY = "ndarray"
    STRUCT_VAR = "var"

    def __init__(self, name, type, shape=None, struct=STRUCT_VAR):
        self.name = name
        self.type = type
        self.shape = shape
        self.struct = struct

    @classmethod
    def field_mapping(cls):
        return {"name": "name", "type": "type", "shape": "shape", "struct": "struct"}

    @staticmethod
    def len_construction_args():
        return 4

    def __str__(self):
        return "name: {}, type: {}, shape: {}, struct: {}".format(self.name, self.type, self.shape, self.struct)


class Output(DictSerializable):

    def __init__(self, name, type, shape=None, max_prob_field=None, raw_prob_field=None, label_prob_fields=None):
        """描述Estimator或者Pipeline的输出列。
        Args:
            name: 列名
            type: 类型
            shape (list): 形状
            max_prob_field: 最大概率列
            raw_prob_field: 原始概率列
            label_prob_fields: 标签概率列
        """
        self.name = name
        self.type = type
        self.shape = shape
        self.max_prob_field = max_prob_field
        self.raw_prob_field = raw_prob_field
        self.label_prob_fields = label_prob_fields

    def to_dict(self):
        result_dict = super(Output, self).to_dict()
        result_dict['maxProbField'] = DictSerializable.to_dict_if_not_none(self.max_prob_field)
        result_dict['rawProbField'] = DictSerializable.to_dict_if_not_none(self.raw_prob_field)
        result_dict['labelProbFields'] = DictSerializable.to_dict_if_not_none(self.label_prob_fields)

        return result_dict

    @classmethod
    def field_mapping(cls):
        return {"name": "name", "type": "type", "shape": "shape"}

    @staticmethod
    def len_construction_args():
        return 6

    @classmethod
    def load_from_dict(cls, dict_data):
        f = super(Output, cls).load_from_dict(dict_data)
        f.max_prob_field = Field.load_from_dict(dict_data.get('maxProbField'))
        f.raw_prob_field = Field.load_from_dict(dict_data.get('rawProbField'))
        f.label_prob_fields = Field.load_from_dict_list(dict_data.get('labelProbFields'))

        return f

    def __str__(self):
        return self.name + "," + self.type


class Attachment(DictSerializable):

    def __init__(self, name, type, file_path, created_date_time):
        self.name = name
        self.type = type
        self.file_path = file_path
        self.created_date_time = created_date_time

    @classmethod
    def field_mapping(cls):
        return {
            "name": "name",
            "type": "type",
            "file_path": "filePath",
            "created_date_time": "createdDateTime"
        }

    def to_dict(self):
        return self.member2dict(self.field_mapping())

    @classmethod
    def load_from_dict(cls, dict_data):
        f = Attachment(None, None, None, None)
        f.dict2member(cls.field_mapping(), dict_data)
        return f


class TrainInfo(DictSerializable):

    def __init__(self, train_set_rows, train_set_cols, train_time):
        self.train_set_rows = train_set_rows
        self.train_set_cols = train_set_cols
        self.train_time = train_time

    @classmethod
    def field_mapping(cls):
        return {"train_time": "trainTime",
                "train_set_cols": "trainSetCols",
                "train_set_rows": "trainSetRows",
                }

    @staticmethod
    def len_construction_args():
        return 3


class ChartData(DictSerializable):

    def __init__(self, name, type, data, attachments=None):
        # 画图时显示名称
        self.name = name
        # 可以是 metrics, file
        self.type = type
        # 字典
        self.data = data
        # 对于自定义类型附件里面就是路径: {"path": path}
        self.attachments = attachments

    @classmethod
    def field_mapping(cls):
        return {
            "name": "name",
            "type": "type",
            "data": "data",
            "attachments": "attachments"
        }

    def to_dict(self):
        m = self.field_mapping()
        return self.member2dict(m)

    @classmethod
    def load_from_dict(cls, dict_data):
        cd = ChartData(None, None, None, None)
        cd.dict2member(cls.field_mapping(), dict_data)
        return cd


class PipelineModelEntry(DictSerializable):

    @classmethod
    def field_mapping(cls):
        return {
            "file_name": "fileName",
            "file_type": "fileType",
            "contains_steps": "containsSteps",
        }

    @staticmethod
    def len_construction_args():
        return 3

    def __init__(self, file_name, file_type, contains_steps):
        self.file_name = file_name
        self.file_type = file_type
        self.contains_steps = contains_steps


class LearningType(object):
    MultiClassify = "MULTICLASSIFY"
    BinaryClassify = "BINARYCLASSIFY"
    Regression = "REGRESSION"
    Clustering = "CLUSTERING"
    Unknown = 'UNKNOWN'
    mapping = {
        MultiClassify: "accuracy",
        BinaryClassify: "roc_auc",
        Regression: "r2",
        Clustering: "unknown"
    }


class PKLSerializable(object):
    """标记该类的子类可以被序列化成pkl文件。"""
    pass


class Module(DictSerializable):

    @staticmethod
    def len_construction_args():
        return 4

    def __init__(self, id, name, extra, steps=None):
        self.id = id
        self.name = name
        self.extra = extra
        if steps is None:
            steps = []
        self.steps = steps

    @classmethod
    def field_mapping(cls):
        return {
            "extra": "extra",
            "name": "name",
            "id": "id",
            "steps": "steps"
        }


@six.add_metaclass(abc.ABCMeta)
class BaseOperatorMetaData(object):

    def __init__(self, module_id):
        self.module_id = module_id
