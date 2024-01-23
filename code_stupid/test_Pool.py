import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import pandas as pd
import sys
import random
import time
from multiprocessing import Pool

print(sys.version)


def square(n):
    time.sleep(0.2)
    return n ** 2


if __name__ == "__main__":
    random.seed(1)
    data = [random.randint(0, 1000) for i in range(50)]

    # 对data每个成员求平方

    # 1.for循环方式
    tis1 = time.time()
    res1 = []
    for n in data:
        res1.append(n ** 2)
        time.sleep(0.2)
    print(res1[:10])
    tie1 = time.time()
    print("1time:", (tie1 - tis1) * 1000)

    # 2.[]方式
    tis2 = time.time()
    res2 = [n ** 2 for n in data]
    time.sleep(0.2 * len(data))
    print(res2[:10])
    tie2 = time.time()
    print("2time:", (tie2 - tis2) * 1000)

    # 3.函数式编程
    tis3 = time.time()
    res3 = map(square, data)
    print(list(res3)[:10])
    tie3 = time.time()
    print("3time:", (tie3 - tis3) * 1000)

    # 4.多进程
    tis4 = time.time()
    p = Pool(4)
    res_l = []
    for n in data:
        res = p.apply_async(square, (n,))
        res_l.append(res)
    p.close()
    p.join()
    print([res_l[i].get() for i in range(10)])
    tie4 = time.time()
    print("4time:", (tie4 - tis4) * 1000)

    # 5.pool.map
    tis5 = time.time()
    p_m = Pool(4)
    res5 = list(p_m.map(square, data))
    print(res5[:10])
    tie5 = time.time()
    print("5time:", (tie5 - tis5) * 1000)

# def deal_df(df):
#     # 构建数据集
#     for index, row in df.iterrows():
#         if not pd.isna(row["c"]):
#             if index == 1:
#                 df.drop(index, inplace=True)
#             tag = 1
#         else:
#             # 如果没有标签，设置一个为0的假标签
#             tag = 0
#     return tag, index
#
#
# df = pd.DataFrame([{"a": 1, "c": 2}, {"a": 3, "c": 4}])
# df.to_csv('./1.csv', index=False)
# print(df.loc[1]['a'])
# print(deal_df(df))
# print(df)
# exit()
# p = [i for i in range(len([1, 2, 3]))]
# random.shuffle(p)
# r = random.sample(p, 2)
# print(r)
#
# img = cv2.imdecode(np.fromfile(r'D:\download\WXWork\1688857557438882\Cache\Image\2023-02\原图.png', dtype=np.uint8), 1)
# cv2.imshow('png', img)
# cv2.waitKey()
# print(img.shape)
# exit()
#
#
# def calc_slope(y):
#     n = len(y)
#     if n == 1:
#         raise ValueError('Can\'t compute slope for array of length=1')
#     x_mean = (n + 1) / 2
#     x2_mean = (n + 1) * (2 * n + 1) / 6
#     xy_mean = np.average(y, weights=np.arange(1, n + 1))
#     y_mean = np.mean(y)
#     slope = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean)
#     return slope
#
#
# point_list = [(0, 2.233344793319702), (1, 2.9218475818634033), (2, 2.149766683578491), (3, 2.1347060203552246),
#               (4, 2.124802350997925), (5, 2.2132198810577393), (6, 1.97186279296875), (7, 2.147073268890381),
#               (8, 2.6671128273010254), (9, 2.564263105392456), (10, 1.8090606927871704), (11, 2.123615026473999),
#               (12, 2.3713173866271973)]
# x = [point[0] for point in point_list]
# y = [point[1] for point in point_list]
#
# print(calc_slope(y))
#
# plt.plot(x, y, linewidth=1)
# plt.scatter(x, y)
# plt.title(" series Numbers", fontsize=24)  # 设置图表标题并给坐标轴加上标签
# plt.xlabel("Value", fontsize=14)
# plt.ylabel("Value", fontsize=14)
# # 设置刻度标记的大小
# plt.tick_params(axis='both', which='major', labelsize=10)
#
# plt.show()
