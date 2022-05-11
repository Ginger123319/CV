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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_net = nn.Sequential(
            BnConv(3, 16, 3, 2, 1),
            ResConv(16),
            ResConv(16),
            BnConv(16, 32, 3, 1),
            ResConv(32),

            nn.Conv2d(32, 64, 3, 2, 1),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )
        # self._rnn_net = nn.GRU(64 * 2 * 7, 128, 2, batch_first=True, bidirectional=False)
        self._rnn_net = nn.LSTMCell(64 * 2, 128)
        self._out_net = nn.Sequential(
            nn.Linear(7*128, 40),
            # nn.Softmax(-1)
        )

    def forward(self, x):
        out = self._conv_net(x)
        out = out.reshape(-1, 64 * 2, 7)
        out = out.permute(0, 2, 1)
        # out = out[:, None, :]
        # # out = out.permute(0, 3, 1, 2)
        # out = out.expand(-1, 4, -1)
        # # # out = out.squeeze()
        outputs = []
        for i in range(out.shape[1]):
            hx, cx = self._rnn_net(out[:, i])
            outputs.append(hx)
        out = torch.stack(outputs, dim=1)
        # print("out",out.shape)
        # out, hn = self._rnn_net(out)
        out = out.reshape(-1, 7*128)
        out = self._out_net(out)
        out = out.reshape(-1, 4, 10)
        out = nn.Softmax(dim=2)(out)
        # # print(out.shape)
        return out


if __name__ == '__main__':
    test_data = torch.randn(5, 3, 60, 240)
    net = Net()
    # print(net)
    out1 = net(test_data)
    print(out1.shape)
    # print(torch.argmax(out1, dim=-1))
    # print(hn1)
