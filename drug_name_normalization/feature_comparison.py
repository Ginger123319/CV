import torch


# 相似度比较
# 向量的内积/向量的模之积
# 测试特征中的每一个向量要与特征库的每一个向量求余弦相似度
# 求出最大的余弦相似度，取到特征库中对应向量的索引
def comparison(feature_test, feature_lib):
    return torch.sum((feature_test[:, None] * feature_lib[None]), dim=-1) / torch.norm(feature_test[:, None],
                                                                                       dim=-1) * torch.norm(
        feature_lib[None],
        dim=-1)


if __name__ == '__main__':
    data1 = torch.randn(100, 2048)
    data2 = torch.randn(3864, 2048)
    print(comparison(data1, data2).shape)
