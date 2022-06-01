import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 一层transformer网络
        # d_model就是V的长度
        # 此处也是SNV结构，torch版本过低，无法调整batch_first参数，默认为false
        self._map_layer = nn.Conv1d(6, 16, 3, 1, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4)
        # 多层transformer网络
        self._sub_net = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # self._sub_net = nn.Transformer(d_model=6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        #                                dim_feedforward=512)

        self._output_net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(1, 2, 0)
        out = self._map_layer(x)
        out = out.permute(2, 0, 1)
        out = self._sub_net(out)
        # print(out.shape)
        # out = out.permute(1, 0, 2)
        # out = out.reshape(-1, 5 * 16)
        out = out[-1, :]
        # print(out.shape)
        out = self._output_net(out)
        return out


if __name__ == '__main__':
    # 此处是SNV结构
    text = torch.randn(5, 30, 6)
    # tag = torch.randn(1, 30, 6)
    transformer_encoder = Net()
    y = transformer_encoder(text)

    print(y.shape)
