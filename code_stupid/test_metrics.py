from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np

y_true = np.array([[0, 1, 0, 1],
                   [0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [1, 1, 1, 0],
                   [1, 0, 1, 1]])

y_pred = np.array([[0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 0, 1]])

from sklearn.metrics import accuracy_score

# 绝对匹配精度，预测结果和标签完全一致才为1，其他情况均为0
# 这样会导致精度过低并且波动范围很大
print(accuracy_score(y_true, y_pred))  # 0.4
print(accuracy_score(y_true, y_pred, normalize=False))  # 2
# 预测为正样本（也就是值为1）的标签中真实为正样本的比例；
from sklearn.metrics import precision_score

print("Precision:", precision_score(y_true=y_true, y_pred=y_pred, average='samples'))  # 0.8
# 真实标签中的正样本标签被预测为正样本的比例；
from sklearn.metrics import recall_score

print("Recall:", recall_score(y_true=y_true, y_pred=y_pred, average='samples'))  # 0.7

# F1 = 2 * (precision * recall) / (precision + recall)
print("F1 samples averaging:", (f1_score(y_true, y_pred, average='samples')))

print("Classification report: \n", (classification_report(y_true, y_pred)))
