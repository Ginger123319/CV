import torch
from torch import nn
from att_multi import MultiAttention


# 此处input_dim和output_dim值一样，没写output_dim
# 自注意力版本
class Transformer(nn.Module):

    def __init__(self, head, input_dim):
        super().__init__()

        self._input_dim = input_dim

        self._att = MultiAttention(head, input_dim, input_dim)

        self._fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        y = self._att(x, x, x)
        # 残差
        y = x + y
        y = y.reshape(-1, self._input_dim)
        z = self._fc(y)
        # 残差
        z = y + z
        z = z.reshape(-1, x.shape[1], self._input_dim)

        return z


if __name__ == '__main__':
    # SNV形状
    text = torch.randn(5, 30, 6)

    transformer = Transformer(3, 6)
    y = transformer(text)
    print(y.shape)
