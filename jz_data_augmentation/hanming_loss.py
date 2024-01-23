# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:26:34 2019@author: muli
"""
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score

y_pred = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 0, 1]])
y_true = np.array([[1, 1, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 1]])
h = hamming_loss(y_true, y_pred)
print("汉明损失：", h)
z = zero_one_loss(y_true, y_pred)
print("0-1 损失：", z)
c = coverage_error(y_true, y_pred) - 1  # 减 1原因：看第2个参考链接
print("覆盖误差：", c)
r = label_ranking_loss(y_true, y_pred)
print("排名损失：", r)
a = average_precision_score(y_true, y_pred)
print("平均精度损失：", a)
