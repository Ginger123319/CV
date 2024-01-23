# -*- encoding: utf-8 -*-

import os
import time
from dc_model_repo.base import StepType, ModelFileFormatType
from dc_model_repo.step.base import BaseEstimator
from dc_model_repo.base import BaseOperator

class OnnxDCEstimator(BaseEstimator):

    def __init__(self, operator, algorithm_name, framework, input_cols, target_cols, output_cols, extension=None, **kwargs):
        super(OnnxDCEstimator, self).__init__(input_cols=input_cols, 
                                                algorithm_name=algorithm_name,
                                                target_cols=target_cols,
                                                output_cols=output_cols,
                                                kind=StepType.Estimator,
                                                framework=framework,
                                                model_format=ModelFileFormatType.ONNX,
                                                extension=extension,
                                                **kwargs)
        self.model_path = operator

    def persist_model(self, fs, destination):
        model_path = os.path.join(self.serialize_data_path(destination), 'model.onnx')
        fs.copy(self.model_path, model_path)
        self.model_path = os.path.join(BaseOperator.FILE_DATA, 'model.onnx')

    def prepare(self, step_path, **kwargs):
        from dc_model_repo.base.mr_log import logger
        import onnxruntime as rt
        # 加载模型
        t1 = time.time()
        p_model = step_path + "/data/model.onnx"
        self.model = rt.InferenceSession(p_model)
        t2 = time.time()
        took = round(t2 - t1, 2)
        logger.info("成功加载模型:\n[%s] ,\n耗时 %s(s)." % (str(self.model), took))

    def predict(self, X):
        import numpy as np
        inputs = [i.name for i in self.input_features]
        outputs = [o.name for o in self.target]
        predict_outputs = outputs.copy()
        for o in self.model.get_outputs():
            if o.name == "output_probability":
                outputs.append('output_probability')
        if isinstance(X, (np.ndarray, list)):
            input_feed = {inputs[i]: np.array(X[i]).astype(self.input_features[i].type) for i in range(len(inputs))}
        elif isinstance(X, dict):
            input_feed = {inputs[i]: np.array(X[inputs[i]]).astype(self.input_features[i].type) for i in range(len(inputs))}
        else:
            raise Exception('not support data type {}'.format(type(X)))
        result = self.model.run(outputs, input_feed)
        result_dict = {}
        for i in range(0, len(predict_outputs)):
            result_dict.setdefault(predict_outputs[i], result[i])

        # 返回dataframe
        # result_df = pd.DataFrame(result_dict)
        # # 如果预测结果包含概率，计算最大概率
        # if 'output_probability' in outputs:
        #     i = outputs.index('output_probability')
        #     prob_detail = result[i]
        #     prob_detail_ = []
        #     for d in prob_detail:
        #         prob_detail_.append({'prob_' + str(k): v for k, v in d.items()})
        #     prob_detail = pd.read_json(json.dumps(prob_detail_))
        #     raw_prob = pd.DataFrame({"raw_prob": prob_detail.values.tolist()})
        #     prob = []
        #     for r in prob_detail.values:
        #         prob.append(max(r))
        #     result_df['prob'] = prob
        #     result_df = pd.concat([result_df, prob_detail, raw_prob], axis=1)
        # return result_df

        # 返回 dict
        # 如果预测结果包含概率，计算最大概率
        if 'output_probability' in outputs:
            i = outputs.index('output_probability')
            # raw_prob
            prob_detail = result[i]
            result_dict.setdefault("prob", prob_detail)
        return result_dict

    def transform(self, X):
        return self.predict(X)

    def get_params(self):
        return None
