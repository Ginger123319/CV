import os
import shutil
import torch
import cfg
from drug_net import Net
from dataset import DrugTagData
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


class FeatureTrainer:
    def __init__(self, net):
        self._net = net.to(cfg.device)

        self._train_loader = DataLoader(DrugTagData(), batch_size=cfg.train_batch_size,
                                        shuffle=False)
        self._log = SummaryWriter("./log")

    def __call__(self):
        self._net.eval()
        feature_li = []
        # 解决cuda内存溢出问题，禁用反向传播的求导计算
        with torch.no_grad():
            for _i, _data in enumerate(self._train_loader):
                _data = _data.to(cfg.device)
                feature = self._net(_data)
                # print(feature.shape)
                # exit()
                feature_li.append(feature)
            feature_lib = torch.cat(feature_li, dim=0)
            # print(feature_lib.shape)
            return feature_lib


if __name__ == '__main__':
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    train = FeatureTrainer()
    train()
