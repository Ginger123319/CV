from torch import optim
from torch.optim import lr_scheduler

from torchvision import models

opt = optim.SGD(models.resnet18().parameters(), lr=0.0001, momentum=0.9)
# 学习率调整策略
# 每10个epoch调整一次，将学习率调整为原来的一半
exp_lr_scheduler = lr_scheduler.StepLR(optimizer=opt, step_size=10, gamma=0.5)
