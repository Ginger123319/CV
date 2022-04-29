import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

tag = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
                0., 0., 0., 0.])
out = np.array([1.1180e-06, 1.9937e-03, 2.8397e-05, 1.6107e-03, 1.2475e-04, 4.7644e-03,
                5.7804e-07, 2.7666e-04, 7.1302e-02, 9.9991e-01, 1.5044e-05, 9.9729e-01,
                1.9848e-04, 2.8832e-06, 9.4510e-02, 9.9963e-01, 7.2722e-03, 2.4726e-07,
                3.9037e-04, 4.1925e-06, 1.9840e-03, 6.0010e-02, 3.7018e-01, 4.1306e-03,
                4.3031e-04, 1.5046e-05, 9.9987e-01, 3.5638e-07, 4.1609e-01, 9.9603e-01,
                9.9975e-01, 9.9852e-01, 1.4167e-04, 1.9937e-03, 5.5709e-05, 9.9992e-01,
                1.3133e-04, 1.6645e-04, 5.3202e-04, 4.8167e-04])
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
plt.show()

# 打印混淆矩阵
out_new = np.array(out > 0.5, dtype=np.float32)
print(out_new)
cm = confusion_matrix(tag, out_new)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()
