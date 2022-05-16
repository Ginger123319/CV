import os
import shutil
import torch
from torch.nn import BCELoss
import cfg
from module.stock_net import Net
from stock_dataset import StockData
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self):
        self._net = Net().to(cfg.device)
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
        self._opt = Adam(self._net.parameters())

        self._train_loader = DataLoader(StockData(cfg.data_dir, is_train=True), batch_size=cfg.train_batch_size,
                                        shuffle=True)
        self._test_loader = DataLoader(StockData(cfg.data_dir, is_train=False), batch_size=cfg.test_batch_size,
                                       shuffle=True)

        self._loss_fn = BCELoss()
        self._log = SummaryWriter("./log")

    def __call__(self, *args, **kwargs):
        for _epoch in range(1000):

            self._net.train()
            _sum_loss = 0.
            for _i, (_target, _data) in enumerate(self._train_loader):
                _data = _data.permute(1, 0, 2)
                _target, _data = _target.to(cfg.device), _data.to(cfg.device)
                # print(_data.shape)
                # exit()

                _out = self._net(_data)[:, 0]
                # 输出和标签形状一致，并且元素数据类型均为float32
                _loss = self._loss_fn(_out, _target)

                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()

                _sum_loss += _loss.cpu().detach().item()

            self._log.add_scalar("train_loss", _sum_loss / len(self._train_loader), _epoch)

            torch.save(self._net.state_dict(), cfg.param_path)
            # torch.save(self._opt.state_dict(), "o.pt")
            print(_sum_loss / len(self._train_loader))

            # 开始验证
            self._net.eval()
            _sum_loss, _sum_acc = 0., 0.

            for _i, (_target, _data) in enumerate(self._test_loader):
                _data = _data.permute(1, 0, 2)
                _target, _data = _target.to(cfg.device), _data.to(cfg.device)

                _out = self._net(_data)[:, 0]

                _loss = self._loss_fn(_out, _target)
                _sum_loss += _loss.cpu().detach().item()

                # 求精度
                # print((_out > 0.5).float())
                # print(_target)
                _sum_acc += torch.mean(torch.eq((_out > 0.5).float(), _target).float())
                _sum_acc = _sum_acc.cpu().detach().item()

            self._log.add_scalar("val_loss", _sum_loss / len(self._test_loader), _epoch)
            self._log.add_scalar("val_acc", _sum_acc / len(self._test_loader), _epoch)
            print(_sum_loss / len(self._test_loader))
            print(_sum_acc / len(self._test_loader))


if __name__ == '__main__':
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    train = Trainer()
    train()
