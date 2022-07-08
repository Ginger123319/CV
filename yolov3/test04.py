import math
import torch

# a = 3.9
# b,c = math.modf(a)
# print(b)
# print(c)
ANCHORS_GROUP = {
    13: [[110, 199], [134, 204], [173, 235]],
    26: [[59, 113], [72, 113], [85, 191]],
    52: [[29, 38], [30, 74], [54, 108]],
}
anchors = ANCHORS_GROUP[13]
anchors = torch.Tensor(anchors)
idxs = torch.tensor([[0, 11, 1, 0],
                     [0, 11, 1, 1],
                     [0, 11, 13, 2]])
print(idxs.shape)
print(idxs[:, 3])
a = idxs[:, 3]
print(anchors[a, 0])
out = torch.stack([idxs[0], idxs[1], idxs[2]], dim=0)  # 将置信度坐标和类别按照一轴即列的方向重组堆叠在一起
print(out)
print(idxs[..., 0] == 0)
