import shutil
import torch
import os
# from net import Net
from torch.utils.tensorboard import SummaryWriter
import cfg
from data_deal import CSVDataset
from net_deal import _get_model
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from early_stopping import EarlyStopping
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 采用数据集的方式实现手写数字识别的训练和测试
class EntropySampling(object):
    def __init__(self, predict):
        super().__init__()
        self.predict = predict

    def query(self, n):
        probs = self.predict.predict_probs()
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)
        return uncertainties.sort()[1][:n]


class Predict:
    def __init__(self, label_path, unlabeled_path, input_shape, model_type, num_class):
        # 数据可视化工具使用
        self.writer = SummaryWriter("./log")

        self.test_data = CSVDataset(label_path=label_path, unlabeled_path=unlabeled_path, input_shape=input_shape,
                                    is_train=False)
        print(self.test_data.class_name)

        # 加载训练和验证数据
        self.test_loader = DataLoader(self.test_data, shuffle=False, batch_size=16, num_workers=0)

        # print(len(train_ds), len(val_ds))
        # exit()

        # 创建网络对象
        self.net = _get_model(model_type=model_type, num_class=num_class).to(DEVICE)
        # 加载参数
        if os.path.exists(cfg.param_path):
            try:
                self.net.load_state_dict(torch.load(cfg.param_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(cfg.param_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")

    def predict(self, *args, **kwargs):
        self.net.eval()
        preds = torch.zeros([2, len(self.test_data)], dtype=torch.int64)
        # 测试,输出与标签作比较，求精度
        with torch.no_grad():
            for i, (id, images, tags, idx) in enumerate(self.test_loader):
                img_data = images.to(DEVICE)
                test_out = self.net.forward(img_data)

                outs = torch.argmax(test_out, dim=1)
                # outs = (test_out > 0.5).float()
                # print(outs.shape)
                # print(outs)
                preds[0][idx] = outs.cpu()
                preds[1][idx] = id
        return preds

    def predict_probs(self, *args, **kwargs):
        self.net.eval()
        probs = torch.zeros([len(self.test_data), len(self.test_data.class_name)])
        with torch.no_grad():
            # 测试,输出与标签作比较，求精度
            for i, (id, images, tags, idx) in enumerate(self.test_loader):
                img_data = images.to(DEVICE)
                test_out = self.net.forward(img_data)

                outs = F.softmax(test_out, dim=1)
                # outs = (test_out > 0.5).float()
                # print(outs.shape)
                probs[idx] = outs.cpu()
        return probs

    def get_strategy(self, strategy_name):
        if strategy_name == "EntropySampling":
            return EntropySampling(self)
        else:
            raise Exception("不支持这种格式{}".format(strategy_name))


if __name__ == '__main__':
    input_label_path = 'temp.csv'
    input_unlabeled_path = 'empty.csv'
    result_path = 'result.csv'
    # 删除log文件
    if os.path.exists(r"./log"):
        shutil.rmtree(r"./log")
        print("log is deleted！")

    test = Predict(label_path=input_label_path, unlabeled_path=input_unlabeled_path, input_shape=(56, 56),
                   model_type="Resnet50", num_class=20)
    class_name = test.test_data.class_name
    preds = test.predict()
    # print(preds)
    probs = test.predict_probs()
    # print(probs)
    strategy = test.get_strategy("EntropySampling")
    # print(strategy.query(5))
    hard_sample_list = strategy.query(5)
    # print(248 in hard_sample_list.tolist())
    # exit()
    sample_id = []
    label = []
    isHardSample = []
    count = 0
    for pred, id in zip(preds[0], preds[1]):
        sample_id.append(id.item())
        label.append({"annotations": [{"category_id": class_name[pred]}]})
        if count in hard_sample_list.tolist():
            isHardSample.append(1)
        else:
            isHardSample.append(0)
        count += 1
    result_dict = {"sample_id": sample_id, "label": label, "isHardSample": isHardSample}
    df = pd.DataFrame(result_dict)
    df.to_csv(result_path)
