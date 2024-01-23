from skopt import BayesSearchCV
from dc_model_repo.step.sklearn_step import SKLearnDCTuningEstimator, SKLearnCommonUtil
from dc_model_repo.base import FrameworkType, ModelFileFormatType, LearningType
from dc_model_repo.util import cls_util


class SkoptDCTuningEstimator(SKLearnDCTuningEstimator):
    """把与BayesSearchCV封装成DC的TuningEstimator

    Args:
        operator (object): SKLearn的estimator。
        dfm_model: 如果需要包装成DataFrameMapper, 表示DFM。
        model (object): APS 封装后支持对列处理的模型。
        kind (str): Step的类型，见 `dc_model_repo.base.StepType`
        input_cols (list): Step处理的列。
        algorithm_name (str): 算法名称。
        extension (dict): 扩展信息字段。
    """

    def __init__(self, operator, input_cols, output_col, target_col, algorithm_name=None, extension=None, **kwargs):

        assert isinstance(operator, BayesSearchCV), "传入了不支持的类型"

        if algorithm_name is None:
            algorithm_name = cls_util.get_class_name(operator.estimator)

        self.tuning_estimator = operator  # 记录cv

        learning_type = kwargs.get("learning_type", None)
        if learning_type is None and SKLearnCommonUtil.is_supported_clustering(operator.estimator):
            learning_type = LearningType.Clustering

        # 1. 调用父类构造方法
        super(SKLearnDCTuningEstimator, self).__init__(operator=operator,
                                                       input_cols=input_cols,
                                                       algorithm_name=algorithm_name,
                                                       target_cols=[target_col],
                                                       output_cols=[output_col],
                                                       framework=FrameworkType.SKLearn,
                                                       model_format=ModelFileFormatType.PKL,
                                                       learning_type=learning_type,
                                                       extension=extension)

        # 2. 初始化变量
        self.labels = None  # labels 的值，不是label_col
        self.model_path = 'data/model.pkl'
