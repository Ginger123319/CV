# 猴子吃桃子问题
# 以后每天早上都吃了前一天剩下的一半零一个
# 到第10天早上想再吃时，见只剩下一个桃子了；求第一天共摘了多少
# 第九天就是第十天的个数加上1的2倍；依次类推；需要倒推9次
# 因为涉及到常数项目2，所以只是用递归思路实现，非典型的递归调用方式
# 准确来说应该叫变量迭代法，使用两个变量就将问题解决了

# 第十天的数量是1
# x2 = 1
# for i in range(9, 0, -1):
#     print(i)
#     x1 = 2 * (x2 + 1)
#     x2 = x1
# print(x1)

# 有一分数序列：2/1，3/2，5/3，8/5，13/8，21/13...求出这个数列的前20项之和
# 构造分数序列
# a = 2
# b = 1
# # print(a / b)
# result = a / b
# for i in range(2, 21):
#     a = a + b
#     b = a - b
#     result = result + (a / b)
# print(result)


# 求1+2!+3!+...+20!的和
# 构造阶乘序列
# def get_num(n):
#     if n == 1:
#         return 1
#     else:
#         return n * get_num(n - 1)
#
#
# result = 0
# for i in range(1, 21):
#     result = result + get_num(i)
# print(result)

# 删除列表中的重复元素
# l1 = [3, 1, 3, 1, 4]
# se = set(l1)
# l2 = list(se)
# print(l2)

# 定义函数实现字符串反转 例如：输入str="string"输出'gnirts'
# 切片[start, stop, step]当step为-1时即反转字符串；
# split() 方法语法：
# str.split(str="", num=string.count(str)).
# 参数
# str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
# num -- 分割次数。默认为 -1, 即分隔所有。split是分割字符串，存储到列表
# s = 'string'
# print(s[::-1])

# 代码实现对列表a中的偶数位置的元素进行加3后求和
# li = [-1, 2, -3, 4]
# sums = 0
# for i in li[0::2]:
#     sums = sums + i + 3
# print(sums)
# # 按绝对值进行从小到大排序
# li.sort(key=abs)
# print(li)
#
# alist = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}, {'name': 'c', 'age': 25}]
#
#
# def get_age(a):
#     return a['age']
#
#
# # 调用sort函数会遍历每个元素，每个元素再去执行参数中涉及到的函数
# alist.sort(key=get_age, reverse=True)
# print(alist)

# 将字符串："k:1|k1:2|k2:3|k3:4"
# 处理成 python 字典：{'k':'1', 'k1':'2', 'k2':'3','k3':'4' }
s = "k:1|k1:2|k2:3|k3:4"
li = s.split('|')
# print(li)
di = {}
for i in li:
    print(i.split(':')[0])
    # 字典的定义，就是指定键和对应的值
    di[i.split(':')[0]] = i.split(':')[1]
print(di)
