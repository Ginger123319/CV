import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, cls_num, feature_dim):
        super().__init__()

        self._center = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, f, t):
        _t = t.long()
        _c = self._center[_t]
        _d = torch.sum((f - _c) ** 2, dim=-1) ** 0.5
        _h = torch.histc(t.float(), 10, min=0, max=9)
        _n = _h[_t]
        return torch.sum(_d / _n)


if __name__ == '__main__':
    cen = CenterLoss(2, 2)
    data = torch.Tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]])
    label = torch.Tensor([0, 0, 1, 0, 1])
    print(cen(data, label))

    center = torch.Tensor([[1, 1], [2, 2]])
    # 根据标签取出对应的中心点
    # 两种方法都可以
    # center_exp = center[label.long()]
    center_exp = torch.index_select(center, dim=0, index=label.long())
    print(center_exp)
    # 不同类别的数据个数可能不一致，如果直接将距离相加，导致不同类别的数据向中心点靠近的损失优化粒度不一致
    # 数据量多的类别的距离加和大一些，优化粒度大一些，可能会出现有的类别的数据已经聚拢，有的还是没有收敛
    # 改进措施：分别除以各个类别对应的数据量，各自求均值，使得各个类别数据的损失权重保持一致，保持同步训练
    # 还有一个原因就是距离和的期望相对于距离和本身稳定一些，更容易训练，更快收敛
    # 根据标签统计各个类别的数据量
    count = torch.histc(label, bins=2, min=0, max=1)
    print(count)
    # 取出每个数据对应类别的数据数量
    count_exp = count[label.long()]
    print(count_exp)
    # 数据点减去中心点，求平方
    # 将x和y的平方加起来
    # 开根号；最后除以各个数据对应类别的数据量
    center_loss = torch.div(torch.sqrt(torch.sum(torch.pow(data - center_exp, 2), dim=-1)), count_exp)
    # 最后将所有距离值加起来
    center_loss = torch.sum(center_loss)
    print(center_loss)
