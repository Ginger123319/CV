import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._rnn_net = nn.LSTMCell(3 * 60, 90)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, 240, 3 * 60)
        # x = x.reshape(-1, 3 * 60, 240)
        # x = x.permute(0, 2, 1)
        outputs = []
        for i in range(x.shape[1]):
            hx, cx = self._rnn_net(x[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        out = nn.Sigmoid()(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._rnn_net = nn.LSTMCell(90, 90)
        self._out_net = nn.Sequential(
            nn.Linear(240 * 90, 40),
        )

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            hx, cx = self._rnn_net(x[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        out = out.reshape(-1, 240 * 90)
        out = self._out_net(out)
        out = out.reshape(-1, 4, 10)
        out = nn.Softmax(dim=2)(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_net = Encoder()
        self._rnn_net = Decoder()

    def forward(self, x):
        out = self._conv_net(x)
        out = self._rnn_net(out)
        return out


if __name__ == '__main__':
    test_data = torch.randn(5, 3, 60, 240)
    net = Net()
    # print(net)
    out1 = net(test_data)
    print(out1.shape)
