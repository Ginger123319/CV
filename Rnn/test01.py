import torch
from torch import nn


class Rnn(nn.Module):

    def __init__(self):
        super().__init__()

        self._sub_net = nn.GRU(26, 17, 2, batch_first=True, bidirectional=False)

    def forward(self, x):
        h0 = torch.zeros(2, 3, 17)
        # c0 = torch.zeros(2, 3, 17)
        output, hn= self._sub_net(x, h0)

        return output, hn


if __name__ == '__main__':
    seq = torch.randn(3, 632, 26)

    rnn = Rnn()
    output, hn = rnn(seq)
    print(output.shape, hn)
