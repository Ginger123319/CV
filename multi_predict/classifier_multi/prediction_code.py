import torch
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Predict:
    def __init__(self, test_data, model):
        self.test_data = test_data
        # 加载训练和验证数据
        self.test_loader = DataLoader(self.test_data, shuffle=False, batch_size=16, num_workers=0)
        # 创建网络对象
        self.net = model.to(DEVICE)

    def predict_probs(self):
        self.net.eval()
        probs = torch.zeros([len(self.test_data), len(self.test_data.class_name)])
        with torch.no_grad():
            # 返回预测概率
            for i, (_, images, tags, idx) in enumerate(self.test_loader):
                img_data = images.to(DEVICE)
                test_out = self.net.forward(img_data)
                outs = torch.sigmoid(test_out)
                probs[idx] = outs.cpu()
        return probs
