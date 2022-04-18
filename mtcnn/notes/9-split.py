# split的用法--------以元素切割

a = "000001.jpg    95  71 226 313"

# 例子1------split()：比例2更简单
print(a.split())

# 例子2------split(" ")
print(a.split(" "))
b = a.split(" ")
print(list(filter(bool,b)))