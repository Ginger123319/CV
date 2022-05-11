import os
import shutil
import torch
from torch.nn import MSELoss
import cfg
# from net import Net
from net3 import Net
# from net2 import Net
from dataset import CodeData
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Test:
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

        self._test_loader = DataLoader(CodeData(cfg.test_dir), batch_size=cfg.test_batch_size,
                                       shuffle=True)

        self._loss_fn = MSELoss()
        self._log = SummaryWriter("./log")

    def __call__(self, *args, **kwargs):
        for _epoch in range(1):
            # 开始验证
            self._net.eval()
            _sum_loss, _sum_acc = 0., 0.

            for _i, (_data, _target) in enumerate(self._test_loader):
                _data, _target = _data.to(cfg.device), _target.to(cfg.device)

                _y, _ = self._net(_data)

                _loss = self._loss_fn(_y, _target)
                _sum_loss += _loss.cpu().detach().item()

                # 求精度
                _y = torch.argmax(_y, dim=-1)
                _target = torch.argmax(_target, dim=-1)
                print("输出：{}\n标签：{}".format(_y[0], _target[0]))
                _sum_acc += torch.mean(torch.eq(_y, _target).float())
                _sum_acc = _sum_acc.cpu().detach().item()

            self._log.add_scalar("val_loss", _sum_loss / len(self._test_loader), _epoch)
            self._log.add_scalar("val_acc", _sum_acc / len(self._test_loader), _epoch)
            print("平均损失：{}".format(_sum_loss / len(self._test_loader)))
            print("平均精度：{}".format(_sum_acc / len(self._test_loader)))


if __name__ == '__main__':
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    test = Test()
    test()
