import torch

from torch import nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.cls_net = nn.Sequential(
            nn.Conv1d(22, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),

            nn.Linear(544, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.mask_net = nn.Sequential(
            nn.Conv1d(22, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1, padding=0),

        )

    def forward(self, x):
        x = x.permute(0, 2, 1)

        mask_y = self.mask_net(x)
        cls_y = self.cls_net(x * mask_y)
        return cls_y, mask_y


if __name__ == '__main__':
    x = torch.randn(2, 300, 22)
    net = Net()
    cls_y, mask_y = net(x)
    print(cls_y.shape, mask_y.shape)
