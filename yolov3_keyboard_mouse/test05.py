# # 修改标签中的非法文件名(IMG_xxx)
# save_file = open("data/label_bak.txt", 'w')
# with open("data/label_ori.txt") as f:
#     for i, line in enumerate(f.readlines()):
#         if line.startswith("I"):
#             new_name = str(i + 1).rjust(4, '0') + '.jpg'
#             li = line.split()
#             li[0] = new_name
#             string = " ".join(li)
#             print(string)
#             save_file.write(string)
#             save_file.write("\n")
#         else:
#             save_file.write(line)
# save_file.close()

# 划分训练集和测试集，其实就是划分标签
import random

val_file = open("data/label_val.txt", 'w')
train_file = open("data/label_train.txt", 'w')
with open("data/label_ori.txt") as f:
    label_li = f.readlines()
    length = len(label_li)
    start = 0
    end = 19
    rand_li = []
    # 按顺序每20张图片随机取一张作为测试集数据，共30张数据
    for i in range(length // 20):
        random.seed(i)
        rand_li.append(random.randint(start, end))
        start = (i + 1) * 20
        end = start + 19
    print(rand_li)
    for i, elem in enumerate(label_li):
        # print(elem)
        if i in rand_li:
            val_file.write(elem)
        else:
            train_file.write(elem)

val_file.close()
train_file.close()
