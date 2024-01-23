from collections import OrderedDict

a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
a_set = OrderedDict.fromkeys(a).keys()  # create an ordered set from list a
b_set = OrderedDict.fromkeys(b).keys()  # create an ordered set from list b
result = a_set & b_set  # perform intersection operation
print(result)  # print the ordered result
# output: {3, 4}
