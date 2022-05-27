import torch
from torch import nn
from torch import optim
from CenterLoss import CenterLoss


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self._f_layer = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self._output_layer = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forwad(self, x):
        _f = self._f_layer(x)
        _y = self._output_layer(_f)
        return _y, _f


class Trainer:

    def __init__(self):
        self._net = Net()
        self._opt = optim.Adam(self._net.parameters())

        self._train_dataloader = None

        self._cls_fn = nn.BCELoss()
        self._center_fn = CenterLoss()

    def __call__(self):

        for _epoch in range(1000000000000000):

            for _i, (_data, _target) in enumerate(self._train_dataloader):
                _y, _f = self._net(_data)

                _loss = self._cls_fn(_y, _target) + 0.8 * self._center_fn(_f, _target)

                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()


class Predictor:

    def __init__(self):
        self._net = Net()
        self._net.load_state_dict(torch.load("w.pt"))
        self._net.eval()

    def __call__(self, x):
        _, _f = self._net(x)
        return _f
