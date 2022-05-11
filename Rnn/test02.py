import torch
from torch import nn


class Rnn(nn.Module):

    def __init__(self):
        super().__init__()

        self._rnn_cell = nn.LSTMCell(26, 17)

    def forward(self, x):
        outputs = []
        hx = torch.zeros(3, 17)
        cx = torch.zeros(3, 17)
        for i in range(x.shape[1]):
            hx, cx = self._rnn_cell(x[:, i], (hx, cx))
            outputs.append(hx)
        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    seq = torch.randn(3, 5, 26)

    rnn = Rnn()
    outputs = rnn(seq)
    print(outputs.shape)
