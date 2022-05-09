import os
import shutil

import torch
from torch.nn import MSELoss
import cfg
from net import Net
from dataset import CodeData
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
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

        self._train_loader = DataLoader(CodeData(cfg.train_dir), batch_size=cfg.train_batch_size, shuffle=True)
        self._validate_loader = DataLoader(CodeData(cfg.test_dir), batch_size=cfg.validate_batch_size,
                                           shuffle=True)

        self._loss_fn = MSELoss()
        self._log = SummaryWriter("./log")

    def __call__(self, *args, **kwargs):
        for _epoch in range(1000):

            self._net.train()
            _sum_loss = 0.
            for _i, (_data, _target) in enumerate(self._train_loader):
                _data, _target = _data.to(cfg.device), _target.to(cfg.device)

                _y, _ = self._net(_data)
                # print(_y.shape)
                # print(_target.shape)
                # _y = torch.argmax(_y, dim=-1)
                _loss = self._loss_fn(_y, _target)

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
            for _i, (_data, _target) in enumerate(self._validate_loader):
                _data, _target = _data.to(cfg.device), _target.to(cfg.device)

                _y, _ = self._net(_data)
                _loss = self._loss_fn(_y, _target)
                _sum_loss += _loss.cpu().detach().item()

                # 求精度
                _y = torch.argmax(_y, dim=-1)
                _target = torch.argmax(_target, dim=-1)
                _sum_acc += torch.mean(torch.eq(_y, _target).float())
                _sum_acc = _sum_acc.cpu().detach().item()

            self._log.add_scalar("val_loss", _sum_loss / len(self._validate_loader), _epoch)
            self._log.add_scalar("val_acc", _sum_acc / len(self._validate_loader), _epoch)
            print(_sum_loss / len(self._validate_loader))
            print( _sum_acc / len(self._validate_loader))


if __name__ == '__main__':
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    train = Trainer()
    train()
