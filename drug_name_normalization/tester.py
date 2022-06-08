import os
import shutil
import torch
import cfg
from drug_net import Net
from dataset import DrugData
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from feature_comparison import comparison
from feature_library_generation import FeatureTrainer


class Trainer:
    def __init__(self):
        self._net = Net().to(cfg.device)
        self._featureTrainer = FeatureTrainer()
        # 加载参数
        if os.path.exists(cfg.param_path):
            try:
                self._net.load_state_dict(torch.load(cfg.param_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(cfg.param_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")

        # self._opt = SGD(self._net.parameters(), lr=5e-4)
        # self._opt = Adam(self._net.parameters())

        # self._train_loader = DataLoader(DrugData(cfg.save_path), batch_size=cfg.train_batch_size,
        # shuffle = True)
        self._test_loader = DataLoader(DrugData(cfg.test_path), batch_size=cfg.train_batch_size, shuffle=True)

        self._log = SummaryWriter("./log")

    def __call__(self, *args, **kwargs):
        feature_lib = self._featureTrainer()
        for _epoch in range(1):

            self._net.eval()
            _train_acc_sum = 0.
            feature_li = []
            with torch.no_grad():
                for _i, (_data, _target) in enumerate(self._test_loader):
                    _data = _data.to(cfg.device)
                    feature = self._net(_data)
                    # print(feature.shape)
                    # exit()
                    # 输出和标签形状一致，并且元素数据类型均为float32
                    feature_li.append(feature)
                feature_test = torch.cat(feature_li, dim=0)
            # print(feature_test.shape)
            feature_test = feature_test.detach().cpu()
            feature_lib = feature_lib.detach().cpu()
            _out = comparison(feature_test, feature_lib)
            print(_out)
            # print(feature.shape)
            #     _loss, _out = self._net.get_loss_fun(feature, _target)
            # 计算精度
            _out = torch.argmax(_out, dim=-1)
            print(_out)
            _y = _target
            print(_y)
            # print(torch.mean((out == y).float()))
            _train_acc = torch.mean((_out == _y).float())
            #
            #     _sum_loss += _loss.cpu().detach().item()
            #
            # self._log.add_scalar("train_loss", _sum_loss / len(self._train_loader), _epoch)
            # self._log.add_scalar("train_acc", _train_acc_sum / len(self._train_loader), _epoch)
            #
            # torch.save(self._net.state_dict(), cfg.param_path)
            # # torch.save(self._opt.state_dict(), "o.pt")
            print("epoch--{} acc:{}".format(_epoch, _train_acc))
            # print("epoch--{} loss:{}".format(_epoch, _sum_loss / len(self._train_loader)))


if __name__ == '__main__':
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    train = Trainer()
    train()
