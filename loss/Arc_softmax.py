from torch import nn
import torch
import torch.nn.functional as F


class ArcSoftmax(nn.Module):
    def __init__(self, feature_dim=2, cls_num=10):
        super().__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, cls_num))

    def forward(self, x, s=10, m=0.01):
        # 在V的维度进行标准化
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # print(x_norm.shape,w_norm.shape)
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.arccos(cosa)
        top = torch.exp(s * torch.cos(a + m) * 10)
        down = top + torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(s * cosa * 10)
        arc_softmax = top / down
        return torch.log(arc_softmax)


if __name__ == '__main__':
    arc = ArcSoftmax()
    feature = torch.randn(3, 2)
<<<<<<< HEAD
    print(arc(feature).shape)
=======
    print(arc(feature))
>>>>>>> 8e573c5ca72d22baed81e334b509533fd6d7a85a
