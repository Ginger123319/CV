import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 一层transformer网络
        # d_model就是V的长度
        # 此处也是SNV结构，torch版本过低，无法调整batch_first参数，默认为false
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=2)
        # 多层transformer网络
        self._sub_net = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self._output_net = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self._sub_net(x)
        # print(yy[-1, :].shape)
        # print(y.shape)
        # 取所有批次的最后一条数据y[-1, :]作为输出层的输入
        # 因为自注意力，最后一条输出包含之前输出的信息，可以作为transformer最终输出
        print(y[-1, :].shape)
        return self._output_net(y[-1, :])


if __name__ == '__main__':
    # 此处是SNV结构
    text = torch.randn(10, 3, 4)

    transformer_encoder = Net()
    y = transformer_encoder(text)

    print(y.shape)
