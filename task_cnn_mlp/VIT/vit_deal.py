import torch
from VIT.vit_transformer import Net
from VIT.cnn_net import CnnNet


def vit(feature_map):
    _n, _c, _h, _w = feature_map.shape
    feature = feature_map.reshape(_n, _c, _h * _w)
    # 变为SNV结构的数据，其中V是通道C，图像的话就是3
    feature = feature.permute(2, 0, 1)
    feature = Net(_c)(feature)
    feature = feature.permute(1, 2, 0)
    feature = feature.reshape(_n, _c, _h, _w)
    return feature


# 在原图上进行处理
def original_vit(x):
    _n, _c, _H, _W = x.shape
    _h, _w = _H // 4, _W // 4
    # 切分原图
    feature = x.reshape(_n, _c, 4, _h, 4, _w)
    feature = feature.permute(0, 2, 4, 1, 3, 5)
    # 将多出来的通道合并到N所在的通道上
    feature = feature.reshape(-1, _c, _h, _w)
    # 放入cnn网络提取特征
    feature = CnnNet()(feature)
    # 放入vit网络进行全局相关性处理，qkv都是自身，自注意力
    feature = vit(feature)
    # 再还原为原始的批次
    _n1, _c1, _h1, _w1 = feature.shape
    feature = feature.reshape(_n, 4, 4, _c1, _h1, _w1)
    feature = feature.permute(0, 3, 1, 4, 2, 5)
    feature = feature.reshape(_n, _c1, 4 * _h1, 4 * _w1)
    # 进行后续的卷积操作
    return feature


if __name__ == '__main__':
    data = torch.randn(5, 3, 100, 100)
    print(original_vit(data).shape)
