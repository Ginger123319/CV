str1 = "aabbcc"
if "a" in str1:
    # print(True)
    str2 = str1.replace("a", "a\n")
    # print(str2)
list1 = [8, 22, 36, 16]
dataset = []
for i, elem in enumerate(list1):
    list2 = []
    tag = elem
    # print(elem)
    print(elem)
    if i - 2 < 0:
        list2.extend(list1[0:i] + list1[i + 1:i + 3])
    else:
        list2.extend(list1[i - 2:i] + list1[i + 1:i + 3])
    print(list2)
    while len(list2) < 4:
        if i - 1 <= 0:
            list2.insert(0, 38)
        if i + 2 >= 4:
            list2.insert(37, 39)
    print("after:", list2)
    dataset.append((list2, tag))
print(dataset)
