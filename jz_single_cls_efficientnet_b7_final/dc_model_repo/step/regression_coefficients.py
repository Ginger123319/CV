# -*- encoding: utf-8 -*-

import json

SKLEARN_LINER_MODEL = 'sklearn.linear_model'
CLASSIFIER = "classifier"
REGRESSOR = "regressor"


class Coefficient:
    def __init__(self, variate, variate2, value, type='is'):
        self.variate = variate
        self.type = type
        self.variate2 = variate2
        self.value = value
        self.valueABS = abs(value)


class CoefIntercept:
    def __init__(self, coef_=[], intercept_=None):
        self.coef_ = coef_
        self.intercept_ = intercept_


# 获取对象的完全限定类名
def fullname(o):
    return o.__module__ + "." + o.__class__.__name__


def is_linear_model(o):
    from dc_model_repo.util.sklearn_util import get_best_estimator_if_cv
    o = get_best_estimator_if_cv(o)
    return fullname(o).startswith(SKLEARN_LINER_MODEL) and hasattr(o, 'coef_')


def build_visual_data(liner_estimator, feature_names=None, class_names=None):
    if is_linear_model(liner_estimator):
        from dc_model_repo.util.sklearn_util import get_best_estimator_if_cv
        liner_estimator = get_best_estimator_if_cv(liner_estimator)
        coef_ = getattr(liner_estimator, 'coef_').tolist()
        intercept_ = getattr(liner_estimator, 'intercept_').tolist()
        if not isinstance(intercept_, list):
            intercept_ = [intercept_]
        classes_coef = {}
        if liner_estimator._estimator_type == CLASSIFIER:
            for coef_row in coef_:
                i_1 = coef_.index(coef_row)
                one_class_coef = []
                for c in coef_row:
                    i_2 = coef_row.index(c)
                    one_class_coef.append(Coefficient(variate=feature_names[i_2], variate2=feature_names[i_2],
                                                      value=round(c, 4)))
                classes_coef.setdefault(str(class_names[i_1]), CoefIntercept(one_class_coef, intercept_[i_1]))
        elif liner_estimator._estimator_type == REGRESSOR:
            reg_coef = []
            for coef_row in coef_:
                i_1 = coef_.index(coef_row)
                reg_coef.append(Coefficient(variate=feature_names[i_1], variate2=feature_names[i_1],
                                value=round(coef_row, 4)))
            classes_coef.setdefault(REGRESSOR, CoefIntercept(reg_coef, intercept_[0]))
        else:
            return None
        json_str = json.dumps(classes_coef, default=obj_2_json)
        return json.loads(json_str)
    else:
        return None


def obj_2_json(obj):
    coef_ = []
    for c in obj.coef_:
        coef_.append({
            'variate': c.variate,
            'type': c.type,
            'variate2': c.variate2,
            'value': c.value,
            'valueABS': c.valueABS
        })
    return {
        'coef_': coef_,
        'intercept_': obj.intercept_
    }
