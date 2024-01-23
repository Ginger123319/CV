import torch
import numpy as np


def cal_test_acc(preds):
    len_pred = len(preds)
    return 1.0 * (preds[:, 1] == preds[:, 2]).sum().item() / len_pred


def get_theta(prediction, label):
    theta = 0.99
    # 将 概率，预测的类别和标签类别stack在一起
    # 输入前prediction在dim=1处做了softmax
    data = prediction
    # 只保留四位小数
    data = data.max(1)
    pred = torch.stack([data[0], data[1], label], dim=1)
    # 根据概率列进行降序排序
    index = torch.argsort(pred[:, 0], descending=False)
    pred = pred[index]
    print(pred[:30])
    # 从上到下依次计算精度
    # 可以改成0.01，然后每次增加0.01比例的数据进行计算，一共计算100次精度？？？？
    step = int(0.1 * len(pred))
    print(step)
    for i in range(len(pred) - step - 1):
        accuracy = cal_test_acc(pred[step + 1 + i:])
        print(len(pred[step + 1 + i:]))
        print("test {} acc: {}".format(i, accuracy))
        if accuracy >= 0.80:
            theta = pred[step + i + 1, 0]
            break
    return theta


if __name__ == '__main__':
    torch.manual_seed(120)
    data = torch.randn(100, 3)
    data = torch.softmax(data, dim=1)
    print(data[0].sum())
    label = torch.randint(0, 3, (100,))
    print(get_theta(data, label))
