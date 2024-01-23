import numpy as np


def quant(x, s):
    return [int(e) for e in np.clip(np.round(x * s), -128, 127)]


arr = np.array([-0.75, -0.5, -0.1, 0.1, 0.2, 0.5, 1])
scale = 127 / np.max(np.abs(arr))
print("量化前:[%s]" % ",".join([str(e) for e in arr]))
print("量化后:[%s]" % ",".join([str(e) for e in quant(arr, scale)]))

# 当区间内最大值与其他值的差异很大的时候
arr = np.array([-0.75, -0.5, -0.1, 0.1, 0.2, 0.5, 100])
scale = 127 / np.max(np.abs(arr))
print("量化前:[%s]" % ",".join([str(e) for e in arr]))
print("量化后:[%s]" % ",".join([str(e) for e in quant(arr, scale)]))

# 选择一个合适的量化因子，重新计算S；这就是排除离群点影响的关键
scale = 127 / 0.75
print("量化前:[%s]" % ",".join([str(e) for e in arr]))
print("量化后:[%s]" % ",".join([str(e) for e in quant(arr, scale)]))
