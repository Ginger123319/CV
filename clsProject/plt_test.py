import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

tag1 = np.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1.,
                 1., 1., 0., 1.])
out1 = np.array([1.4186e-05, 8.2433e-06, 9.9999e-01, 5.3563e-04, 1.3252e-05, 8.0030e-06,
                 7.2895e-01, 9.9796e-01, 3.9677e-05, 4.1905e-01, 2.4475e-07, 7.1064e-09,
                 2.3103e-04, 2.1460e-04, 9.9917e-01, 9.3939e-08, 5.3555e-01, 1.8567e-01,
                 5.8331e-04, 1.5527e-04, 8.0218e-08, 4.5240e-04, 5.1889e-04, 1.3914e-05,
                 4.2669e-09, 1.2296e-08, 4.8318e-09, 4.2470e-10, 1.1918e-06, 7.1794e-01,
                 2.8557e-04, 4.5240e-04, 2.4377e-04, 6.0067e-04, 9.9920e-01, 9.9999e-01,
                 9.7455e-01, 9.9822e-01, 2.1483e-04, 9.9993e-01])
tag = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1.])
out = np.array([1.6784e-04, 1.8063e-07, 4.9012e-05, 6.1854e-10, 9.0334e-05, 6.0501e-06,
                3.0243e-03, 4.9012e-05, 4.7180e-05, 4.1347e-04, 5.9574e-04, 6.6229e-03,
                1.1203e-07, 4.4023e-04, 1.2514e-08, 6.8454e-08, 1.1760e-08, 5.8841e-09,
                4.9012e-05, 1.2141e-07, 9.9872e-01, 9.9860e-01, 1.2425e-06, 9.9974e-01,
                8.9838e-01, 9.9927e-01, 9.9837e-01, 9.3305e-01, 9.3369e-01, 9.9898e-01,
                9.3313e-01, 9.6667e-01, 7.9615e-01, 9.9929e-01, 9.9678e-01, 9.9006e-01,
                9.9914e-01, 9.9766e-01, 9.9825e-01, 2.2135e-05])
plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(tag, out, pos_label=1)
plt.subplot(121)
plt.subplots_adjust(wspace=0.35)
# x = np.arange(0, 1.2, 0.2)
# y = np.arange(0, 1.2, 0.2)
# plt.xticks(x)
# plt.yticks(y)
plt.xlabel('(FPR)')
plt.ylabel('(TPR)')
plt.title('ROC')
plt.plot(fpr, tpr, marker='o')
# plt.show()
AUC = auc(fpr, tpr)
AUC = roc_auc_score(tag, out)
print(AUC)

# plt.figure()
precision, recall, thresholds = precision_recall_curve(tag, out)
plt.subplot(122)
# x = np.arange(0, 1.0, 0.2)
# y = np.arange(0, 1.25, 0.25)
# plt.xticks(x)
# plt.yticks(y)
plt.xlabel('(R)')
plt.ylabel('(P)')
plt.title('PR')
plt.plot(recall, precision)
# print(recall, precision)
plt.show()

# 打印混淆矩阵
out_new = np.array(out > 0.5, dtype=np.float32)
# print(out_new)
cm = confusion_matrix(tag, out_new)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()
