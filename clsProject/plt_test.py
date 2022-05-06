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
out = np.array([2.2989e-04, 1.4274e-03, 1.4814e-03, 1.1943e-02, 2.0123e-04, 1.0373e-03,
                2.7128e-04, 1.4814e-03, 1.7206e-03, 6.9556e-03, 1.6017e-03, 1.2850e-03,
                4.1316e-04, 5.2233e-06, 1.8059e-05, 2.0328e-03, 2.8196e-05, 1.8251e-03,
                1.4814e-03, 5.0612e-04, 9.9897e-01, 9.9697e-01, 1.1793e-03, 9.9893e-01,
                5.2636e-01, 9.9492e-01, 9.9777e-01, 5.7597e-01, 6.2491e-01, 9.9779e-01,
                6.2680e-01, 9.7236e-01, 6.2900e-02, 9.9490e-01, 9.7253e-01, 7.9731e-01,
                9.9824e-01, 9.9819e-01, 9.9825e-01, 2.1945e-04])
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
