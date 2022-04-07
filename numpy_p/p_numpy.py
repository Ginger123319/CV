import numpy as np

# 列表与数组互换
li = [[1, 2], [3, 4], [4, 5]]
a = np.array(li)
# <class 'numpy.ndarray'>
print(type(a))

print(a)
b = a.tolist()
print(b)
# <class 'list'>
print(type(b))

c = np.ndarray([12]).reshape(2, 2, 3)
print(c)

print(np.shape(c))
# 查看ndarray的属性
print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype)
print(a.astype(float))

# print(e)
f = np.ones(shape=[4, 10], dtype=np.float32)
# print(f)
# 只有前面有打印数组，空矩阵才会和打印的数组一样，如果前面只是定义而没有打印，还是一个随机矩阵
g = np.empty(shape=[4, 10], dtype=np.float32)
print(g)

# 生成零数组、一数组、空数组
# one-hot模型，训练矩阵
e = np.zeros(shape=[4, 10], dtype=np.float32)
print(e)
# 先用循环进行处理
# 枚举，第一个参数i遍历索引值，第二个参数k遍历每个元素（此处为列表数据）
for i, k in enumerate(e):
    # print(i)
    # print(k)
    if i == 0:
        k[8] = 8
    elif i == 1:
        k[5] = 5
    elif i == 2:
        k[2] = 2
    else:
        k[0] = 1
print(e)
# axis=0【纵向比较】元素按照列进行比较大小，每列出一个元素；
# axis=1【横向比较】每一行元素进行比较，取出相应的最值
# 函数中带有arg的取索引值；横向比较取元素的横坐标；纵向比较取元素的纵坐标
# 三维数组并不适用，需要另外进行讨论
f = np.argmax(e, axis=1)
print(f)

# 两行两列每个位置存储的是一个一维数组（含三个元素），即[ 0  1  2]；[ 3  4  5]
# [ 6  7  8]；[ 9 10 11]；这就是三维数组，也是三阶张量
d = np.arange(12, dtype=np.float32).reshape(4, 3)
print(d)
# 最小值坐标列表
g = np.argmin(d, axis=1)
# 取每行中最小的值
g = np.argmin(d, axis=1)
# 取每列中最大的值
h = np.argmax(d, axis=1)
# print(g)
# print(h)

# 三维数组中轴的使用说明
# 0，1，2轴：0轴有2个元素，1轴有3个元素，2轴有2个元素
i = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
print(i)
print(np.argmax(i, axis=2))
