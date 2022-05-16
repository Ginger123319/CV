import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 一层transformer网络
        # d_model就是V的长度
        # 此处也是SNV结构，torch版本过低，无法调整batch_first参数，默认为false
        encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=2)
        # 多层transformer网络
        self._sub_net = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self._output_net = nn.Sequential(
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self._sub_net(x)
        # print(out.shape)
        # out = out.permute(1, 0, 2)
        # out = out.reshape(-1, 5 * 6)
        out = out[-1, :]
        return self._output_net(out)


if __name__ == '__main__':
    # 此处是SNV结构
    text = torch.randn(5, 30, 6)
    transformer_encoder = Net()
    y = transformer_encoder(text)

    print(y.shape)
