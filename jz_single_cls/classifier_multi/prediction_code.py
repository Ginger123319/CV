import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Predict:
    def __init__(self, test_data, model):

        self.test_data = test_data

        # 加载训练和验证数据
        self.test_loader = DataLoader(self.test_data, shuffle=False, batch_size=16, num_workers=0)

        # 创建网络对象
        self.net = model.to(DEVICE)

    def predict(self):
        self.net.eval()
        preds = torch.zeros([2, len(self.test_data)], dtype=torch.int64)
        # 返回预测类别和sampleId
        with torch.no_grad():
            for i, (sample_id, images, tags, idx) in enumerate(self.test_loader):
                img_data = images.to(DEVICE)
                test_out = self.net.forward(img_data)

                outs = torch.argmax(test_out, dim=1)

                preds[0][idx] = outs.cpu()
                preds[1][idx] = sample_id
        return preds

    def predict_probs(self):
        self.net.eval()
        probs = torch.zeros([len(self.test_data), len(self.test_data.class_name)])
        with torch.no_grad():
            # 返回预测概率
            for i, (_, images, tags, idx) in enumerate(self.test_loader):
                img_data = images.to(DEVICE)
                test_out = self.net.forward(img_data)
                outs = F.softmax(test_out, dim=1)
                probs[idx] = outs.cpu()
        return probs

    def get_strategy(self, strategy_name, query_num):
        if query_num > len(self.test_data):
            query_num = len(self.test_data)
        if strategy_name == "EntropySampling":
            probs = self.predict_probs()
            log_probs = torch.log(probs)
            uncertainties = (probs * log_probs).sum(1)
            return uncertainties.sort()[1][:query_num]
        elif strategy_name == "LeastConfidence":
            probs = self.predict_probs()
            uncertainties = probs.max(1)[0]
            return uncertainties.sort()[1][:query_num]
        elif strategy_name == "MarginSampling":
            probs = self.predict_probs()
            probs_sorted, idxs = probs.sort(descending=True)
            uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
            return uncertainties.sort()[1][:query_num]
        elif strategy_name == "RandomSampling":
            return random.sample([i for i in range(len(self.test_data))], query_num)
        else:
            raise Exception("不支持这种格式: {} 的查询策略".format(strategy_name))
