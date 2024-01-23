# -*- encoding: utf-8 -*-


def get_best_estimator_if_cv(o):
    if is_cv(o):
        return o.best_estimator_
    else:
        return o


def is_cv(o):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    return isinstance(o, GridSearchCV) or isinstance(o, RandomizedSearchCV)
