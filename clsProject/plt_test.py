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
out = np.array([3.8239e-03, 4.3674e-05, 9.4047e-03, 4.7745e-06, 2.6181e-03, 3.9661e-05,
                1.3101e-01, 9.4047e-03, 1.6436e-02, 1.2450e-01, 9.2069e-02, 7.2941e-02,
                5.5767e-06, 2.3131e-02, 2.4661e-06, 8.5190e-05, 3.2477e-05, 3.0605e-04,
                9.4047e-03, 6.1136e-05, 9.9913e-01, 9.9931e-01, 2.5546e-04, 9.9972e-01,
                8.8885e-01, 9.9832e-01, 9.9787e-01, 7.1925e-01, 7.6405e-01, 9.9850e-01,
                8.8312e-01, 9.7617e-01, 9.7216e-01, 9.9818e-01, 9.9709e-01, 9.9295e-01,
                9.9857e-01, 9.9720e-01, 9.9814e-01, 6.9823e-03])
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
