import torch
from torch import nn


class Net(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        # 一层transformer网络
        # d_model就是V的长度
        # 此处也是SNV结构，torch版本过低，无法调整batch_first参数，默认为false
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        # 多层transformer网络
        self._sub_net = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, feature_map):
        _n, _c, _h, _w = feature_map.shape
        feature = feature_map.reshape(_n, _c, _h * _w)
        # 变为SNV结构的数据，其中V是通道C，图像的话就是3
        feature = feature.permute(2, 0, 1)
        feature = self._sub_net(feature)
        feature = feature.permute(1, 2, 0)
        feature = feature.reshape(_n, _c, _h, _w)
        return feature


if __name__ == '__main__':
    # 此处是SNV结构
    # text = torch.randn(10, 300, 4)
    text = torch.randn(48, 16, 8, 8)

    transformer_encoder = Net(16)
    y = transformer_encoder(text)

    print(y.shape)
