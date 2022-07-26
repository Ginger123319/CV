import torch
from torch import nn


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        """
            input (n_batch, n_channel, n_frame, h, w) = (n_batch, 3, 16, 240, 320)
        """
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 8 * 11, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2),
            nn.ReLU()
            # nn.CrossEntropyLoss() realize the softmax layer
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    data = torch.randn(1, 3, 20, 240, 320)
    net = C3D()
    print(net(data).shape)
