import torch

data = torch.tensor([
    [1, 2, 3],
    [2, 3, 0]
])

src = torch.tensor([  # N
    [  # C
        # V
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    [  # C
        # V
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
])

mask = torch.tensor([
    [  # C
        [1, 0, 1],
    ],
    [
        [0, 1, 0]
    ]

])

# print(src.shape, mask.shape)
#
# print(src * mask)

# print(mask == 1)
# print(torch.nonzero(mask == 1))
# print(data.shape, mask[:, 0].shape)

for d, m in zip(data, mask[:, 0]):
    print(m == 1)
    print(d[m == 1])

# print([d[m == 1] for d, m in zip(data, mask[:, 0])])
