import colorsys

import numpy as np
import torch


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    # 切片，反向输出::-1
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


if __name__ == '__main__':
    path = r"./anchors.txt"
    anchors = get_anchors(path)
    print(anchors)
    print(len(anchors))
    print(len(anchors[0]))
    for w, h in anchors.reshape(-1, 2):
        print(w, h)

    print(torch.floor(torch.tensor(3.2)))
    t1 = torch.Tensor(3, 5)
    # print(t1[..., 2:3][0])
    print(t1[..., 0][[0, 1]])
    t2 = torch.zeros_like(t1)
    # print(t2)
    t3 = torch.cat([torch.zeros_like(t1), torch.zeros_like(t2), t1, t2], 1)
    # print(t3.shape)
    print(torch.linspace(0, 12, 13).repeat(13, 1).repeat(30, 1, 1).shape)
    x = torch.tensor(3)
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    print(t1.index_select(1, LongTensor([0])))
    print(111111111111111111111111)
    print(t1)
    # dim=0,就是每一个张量的对应位置比较大小；最终将0维变成1（或者叫消失）
    print(torch.max(t1, dim=0))
    print(t1[2].shape)


class Color(object):
    def __init__(self):
        # 画框设置不同的颜色
        self.class_names = ["A", "B"]
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def get_color(self):
        return self.colors


print(Color().get_color())
print([1, 2] * 2)
output = [None for _ in range(len([1, 2]))]
print(output)
image_pred = t1
class_conf = torch.randn(3, 1)
conf_thres = 0.5
conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
print(conf_mask)
print(image_pred[conf_mask])
