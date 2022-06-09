import torch
from torch import nn

from Arc_softmax import ArcSoftmax


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.ce_loss = nn.NLLLoss()
        self.arc_loss = ArcSoftmax(128, 3864)
        # 一层transformer网络
        # d_model就是V的长度
        # 此处也是SNV结构，torch版本过低，无法调整batch_first参数，默认为false
        self._map_layer = nn.Sequential(
            nn.Conv1d(300, 384, 3, 2, 1, bias=False),
            nn.BatchNorm1d(384),
            nn.Hardswish(),
            nn.Conv1d(384, 512, 3, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.Hardswish()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=128, activation="gelu")
        # 多层transformer网络
        self._sub_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self._sub_net = nn.Transformer(d_model=6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        #                                dim_feedforward=512)

        self._output_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Linear(256, 128)
        )

    def get_loss_fun(self, feature, labels):
        outputs = self.arc_loss(feature)
        ce_loss = self.ce_loss(outputs, labels)
        return ce_loss, outputs

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self._map_layer(x)
        out = out.permute(2, 0, 1)
        out = self._sub_net(out)
        # print(out.shape)
        out = out.permute(1, 0, 2)
        # out = out.reshape(-1, 8 * 512)
        out = out[:, -1]
        # print(out.shape)
        out = self._output_net(out)
        return out


if __name__ == '__main__':
    # 此处是SNV结构
    text = torch.randn(5, 57, 300)

    # tag = torch.randn(1, 30, 6)
    transformer_encoder = Net()
    y = transformer_encoder(text)
    print(y.shape)
