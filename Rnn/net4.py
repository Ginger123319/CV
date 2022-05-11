import torch
from torch import nn


class BnConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ResConv(nn.Module):

    def __init__(self, in_channels, multiple=0.5):
        super().__init__()

        m_channels = int(in_channels * multiple)
        # print(m_channels)
        self._sub_net = nn.Sequential(
            BnConv(in_channels, m_channels, 1),
            BnConv(m_channels, m_channels, 3, 2, 1, m_channels),
            BnConv(m_channels, in_channels, 1),
        )

    def forward(self, x):
        return self._sub_net(x)


class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_net = nn.Sequential(
            BnConv(3, 16, 3, 2, 1),
            ResConv(16),
            ResConv(16),
            BnConv(16, 32, 3, 1),
            ResConv(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._conv_net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._rnn_net = nn.LSTMCell(3 * 60, 180)

    def forward(self, x):
        x = x.permute(0, -1, 1, 2)
        x = x.reshape(-1, 240, 3 * 60)
        # x = x[:, None, :]
        # x = x.expand(-1, 4, -1)
        outputs = []
        for i in range(x.shape[1]):
            hx, cx = self._rnn_net(x[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        out = nn.Softmax(dim=2)(out)
        return out


class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self._rnn_net = nn.LSTMCell(64 * 2 * 7, 128)
        self._out_net = nn.Sequential(
            nn.Linear(4 * 128, 40),
        )

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            hx, cx = self._rnn_net(x[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        out = out.reshape(-1, 4 * 128)
        out = self._out_net(out)
        out = out.reshape(-1, 4, 10)
        out = nn.Softmax(dim=2)(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._rnn_net = nn.LSTMCell(180, 180)
        self._out_net = nn.Sequential(
            nn.Linear(240 * 180, 40),
        )

    def forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            hx, cx = self._rnn_net(x[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        out = out.reshape(-1, 240 * 180)
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
        # out = out.reshape(-1, 64 * 2 * 7)
        # out = out[:, None, :]
        # out = out.expand(-1, 4, -1)
        out = self._rnn_net(out)
        return out


if __name__ == '__main__':
    test_data = torch.randn(5, 3, 60, 240)
    net = Net()
    # print(net)
    out1 = net(test_data)
    print(out1.shape)
    # print(torch.argmax(out1, dim=-1))
    # print(hn1)
