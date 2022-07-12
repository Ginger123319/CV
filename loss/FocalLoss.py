import torch
from torch import nn


class FocalBCE(nn.Module):
    # alpha调控样本均衡,beta控制样本的训练权重,困难样本的权重加大
    def __init__(self, alpha=0.25, beta=1.5):
        super().__init__()

        self._alpha = alpha
        self._beta = beta

    def forward(self, y, t):
        _p_p = y[t == 1]

        _n_p = y[t == 0]

        _p_loss = -self._alpha * (1 - _p_p) ** self._beta * torch.log(_p_p)
        _n_loss = -(1 - self._alpha) * _n_p ** self._beta * torch.log(1 - _n_p)

        # _loss = torch.mean(_p_loss) + torch.mean(_n_loss)
        _loss = torch.mean(torch.cat([_p_loss, _n_loss], dim=-1))
        return _loss


class FocalCE(nn.Module):

    def __init__(self, alpha=0.25, beta=1.5):
        super().__init__()

        self._alpha = alpha
        self._beta = beta

    def forward(self, x, t):
        _p_p = x[t == 1]

        _loss = -torch.mean(self._alpha * (1 - _p_p) ** self._beta * torch.log(_p_p))
        return _loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
        print(self.reduction)

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        print(loss)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        print(pred_prob)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        print(p_t)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        print(alpha_factor)
        modulating_factor = (1.0 - p_t) ** self.gamma
        print(modulating_factor)
        loss *= alpha_factor * modulating_factor
        print(loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


if __name__ == '__main__':
    loss_fun = nn.BCEWithLogitsLoss()
    pred = torch.tensor([0.2, 0.7, 0.1])
    true = torch.tensor([0., 1., 0])
    print(loss_fun(pred, true))
    # print(FocalBCE()(pred, true))
    # print(FocalCE()(pred, true))
    print(FocalLoss(loss_fun)(pred, true))
