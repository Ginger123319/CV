import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'
    # 实验中, gamma=2的效果最好

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        
        # 专注参数（focusing parameter）：调节易分样本的权重，从而使得模型在训练时更专注于难分类的样本
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
