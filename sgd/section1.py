# # 不求甚解，由浅入深
# # 整体概念，遇到卡壳的卡一会就往下走，跟上整体节奏，然后再回来解决
#
# """
# 这是多行注释，使用双引号。
# 这是多行注释，使用双引号。
# 这是多行注释，使用双引号。
# """
# print(1)
# a = 1
# b = 2
# print(a, b)
# # 代码块
# if True:
#     a = 3
#     print(a)
#
#
# # 类
# class Person:
#     # 默认行为，比如类的构造函数
#     # 属性
#     def __init__(self, name, eye):
#         self.name = name
#         self.eye = eye
#         print("initializing...")
#
#     def __call__(self, *args, **kwargs):
#         print("调用call行为，AI中很常见！")
#
#     # 成员函数
#     def say(self, content, answer):
#         print(content)
#         return answer
#
#     def show(self):
#         print(self.name)
#         self.eye.show()
#
#
# # 实例化-对象
# # p1 = Person("Peter")
# # p2 = Person("Alice")
# # print(p1.say("hello", "me too"))
# # p2.say("pardon", "nothing")
# # # 对象使用()即可调用call行为
# # p2()
# # p2.show()
#
#
# # 继承属性和行为
# class Animal:
#     def __init__(self, name):
#         self.name = name
#
#     def show(self):
#         print(self.name)
#
#
# class Cat(Animal):
#     def __init__(self, name):
#         super(Cat, self).__init__(name)
#
#
# cat = Cat("miao")
# cat.show()
#
#
# # 和Person类是组合关系
# class Eye:
#     def __init__(self, color):
#         self.color = color
#
#     def show(self):
#         print(self.color)
#
#
# eye = Eye("blue")
# p3 = Person("Alice", eye)
# p3.show()
#
# # 面向过程编程
# li = [[1, 2], [1, 3]]
# tu = (1, 2, 3)
# se = {1, 2, 3}
# print("===list操作===")
# print(len(li))
# print(li[0])
# li[1] = 1
# print(li)
#
# li.remove(li[0])
# print(li)
#
# li.append([1, 2])
# print(li)
# # 在第一个位置处插入数字7
# li.insert(1, 7)
# print(li)
#
# li.extend([0, 2, 5])
# print(li)
#
# print(li.index([1, 2]))
#
# li.remove([1, 2])
# print(li)
#
# li.sort()
# print(li)
#
# li.clear()
# print(li)
#
# li.extend([[2, 1], [1, 2]])
# print(li)
#
#
# def get_num(elem):
#     return elem[0]
#
#
# # 如果列表存储的不是基本数据类型，需要指定key即按照同一个标准进行排序
# # 排序默认是从小到大排序
# # li.sort(key=get_num)
# # x就是入参，此处就是列表中的每一个元素，取列表中每一个元素的第一个元素作为排序标准
# li.sort(key=lambda x: x[0])
# print(f"sorted list is {li}")
#
# # 列表切片 list[start:stop:step];step为-1时为反向切片
# # 切片会生成一个新的列表
# print(li[0])
# print(li[::-1])
# print(li[:1])
#
# # 元组操作
# print(tu)
# print(len(tu))
#
# print(tu.index(2))
#
# # 不允许修改元组的值以及顺序，所以所有试图改变元组的操作都会报错
# # TypeError: 'tuple' object does not support item assignment
# # tu[2] = 1
#
# # set操作
# print(se)
# print(len(se))
#
# # set是无序的，所以任何涉及到索引的操作都不能进行，并且set中的元素并不会重复
# # TypeError: 'set' object is not subscriptable【无索引的】
# se.add(4)
# print(se)
# se.add(5)
# print(se)
#
# # 字典操作
# di = {}
# print(type(di))
# di["hello"] = 5
# print(di)
# di["nick"] = 4
# print(di)
# di.pop("nick")
# di["hello"] = 4
# print(di)
#
# # 转义字符
# print("hello\nworld")
# # r字符串可以让转义字符失效，保持原样
# print(r"hello\nworld")
# s = "hello"
# print(s[3::-1])
# print(s.split())
# print(list(s))
#
# print(s.index("o"))
# print(s.find("l"))
# # ValueError: substring not found
# # print(s.index("ol"))
# # 若字串不存在则返回-1
# print(s.find("ol"))
#
# # 格式化字符串
# s1 = "you"
# print(f"{s} {s1}")
#
# # 循环语句
# for i in range(9):
#     print(i, end=" ")
# print()
# for l in li:
#     print(l, end=" ")
# print()
# # k就是列表的索引
# for k, v in enumerate(li):
#     print(k)
#     print(v)
# # 遍历列表，字典对象中取出key和value
# # dict_items([('hello', 4)])
# di["pretty"] = "girl"
# print(di.items())
# for k, v in di.items():
#     print(k, ":", v)
#
# # continue跳过本次循环
# # break 整个循环终止，不再执行
#
# # 列表推导式，简化对列表的一些基本操作，比如每个元素乘以2
# # lambda表达式与之不同，注意区分开
# lis = [1, 2, 3]
# a_list = [x * 2 for x in lis]
# print(a_list)


# 函数，如果不需要参数时调用函数不需要加上括号，成员函数是因为要传入对象本身即self
# 模块 import导入即可
# main函数作用就是保证调用函数的时候不会执行导入文件中执行的结果，不会出现歧义
# 有main函数的文件，只有在执行本文件时，文件中的代码才会被执行

# 定义矩阵并完成矩阵的加法
# 运算符重载
class Matrix:

    # 此处data就是一个嵌套列表，形如[[2, 1], [1, 2]]
    def __init__(self, data):
        self.data = data

    # 重载加号运算
    def __add__(self, other):
        a_row = len(self.data)
        a_col = len(self.data[0])

        b_row = len(other.data)
        b_col = len(other.data[0])

        if a_row != b_row and a_col != b_col:
            return "矩阵形状不同无法进行相加！"
        # 进入加法程序
        for row in range(a_row):
            for col in range(a_col):
                self.data[row][col] += other.data[row][col]
        # 返回值就是当前对象
        return self

    # 重载打印语句,不进行重载时打印的是当前对象的存储地址
    def __str__(self):
        return str(self.data)


if __name__ == '__main__':
    a = Matrix([[2, 1], [1, 2]])
    b = Matrix([[2, 1], [1, 2]])
    print(a)
    print(b)
    print(a + b)
