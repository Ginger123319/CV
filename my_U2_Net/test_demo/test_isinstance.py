# 当我们定义一个class的时候，我们实际上就定义了一种数据类型。
# 我们定义的数据类型和Python自带的数据类型，比如str、list、dict没什么两样：
# 判断一个变量是否是某个类型可以用isinstance()判断：
class Student():
    def __init__(self, name, score):
        self.name = name
        self.score = score

a = '10'
b = 3
c = [1, 2, 3]
d = (1, 2, 3)
f = Student('Eden', 99.9)

print(isinstance(a,  str))      # True
print(isinstance(b, int))       # True
print(isinstance(c, list))      # True
print(isinstance(d, tuple))     # True
print(isinstance(f, Student))   # True
print(isinstance(a, int))   # False
print(isinstance(b, str))   # False
