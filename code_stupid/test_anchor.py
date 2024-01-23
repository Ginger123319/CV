import numpy as np


# --------------------------------------------#
#   生成基础的先验框
# --------------------------------------------#
def generate_anchor_base(base_size=16, ratios=None, anchor_scales=None):
    if anchor_scales is None:
        anchor_scales = [8, 16, 32]
    if ratios is None:
        ratios = [0.5, 1, 2]
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


if __name__ == '__main__':
    nine_anchors = generate_anchor_base()
    print(nine_anchors)
