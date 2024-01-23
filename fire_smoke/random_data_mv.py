import os
import shutil

saveBasePath = "../../source/yolodata/smoke/main"

with open(os.path.join(saveBasePath, 'train.txt')) as f:
    train_number = f.readlines()

with open(os.path.join(saveBasePath, 'val.txt')) as f:
    val_number = f.readlines()

with open(os.path.join(saveBasePath, 'test.txt')) as f:
    test_number = f.readlines()
print("打开文件结束！")
train_image = []
val_image = []
test_image = []

train_txt = []
val_txt = []
test_txt = []

for i in train_number:
    i = i.rstrip('\n')
    i_image = i + '.jpg'
    i_txt = i + '.txt'
    # print(i_image)
    train_txt.append(i_txt)
    train_image.append(i_image)
# print(train_image)

for i in val_number:
    i = i.rstrip('\n')
    i_image = i + '.jpg'
    i_txt = i + '.txt'
    # print(i_image)
    val_txt.append(i_txt)
    val_image.append(i_image)
# print(val_image)

for i in test_number:
    i = i.rstrip('\n')
    i_image = i + '.jpg'
    i_txt = i + '.txt'
    # print(i_image)
    test_txt.append(i_txt)
    test_image.append(i_image)
# print(test_image)
print("图片标签分配完毕！")
# pic mv
new_train = "../../source/yolodata/smoke/images/train"
if not os.path.exists(new_train):
    os.makedirs(new_train)
new_val = "../../source/yolodata/smoke/images/val/"
if not os.path.exists(new_val):
    os.makedirs(new_val)
new_test = "../../source/yolodata/smoke/images/test/"
if not os.path.exists(new_test):
    os.makedirs(new_test)
print("图片目录创建完毕！")
dir_path = "../../source/fire_smoke/data/smoke_20220816/smoke-all-imgs-0415"
for i in train_image:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == i:
                shutil.copy(os.path.join(root, file), new_train)

for i in val_image:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == i:
                shutil.copy(os.path.join(root, file), new_val)

for i in test_image:
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file == i:
                shutil.copy(os.path.join(root, file), new_test)
print("图片分配完毕！")
# label mv
new_train = "../../source/yolodata/smoke/labels/train"
if not os.path.exists(new_train):
    os.makedirs(new_train)
new_val = "../../source/yolodata/smoke/labels/val/"
if not os.path.exists(new_val):
    os.makedirs(new_val)
new_test = "../../source/yolodata/smoke/labels/test/"
if not os.path.exists(new_test):
    os.makedirs(new_test)
print("标签目录创建完毕！")
label_path = "../../source/yolodata/smoke/labels/train"
# for i in train_txt:
#     for root, dirs, files in os.walk(label_path):
#         for file in files:
#             if file == i:
#                 shutil.move(os.path.join(root, file), new_train)

for i in val_txt:
    for root, dirs, files in os.walk(label_path):
        for file in files:
            if file == i:
                shutil.move(os.path.join(root, file), new_val)

for i in test_txt:
    for root, dirs, files in os.walk(label_path):
        for file in files:
            if file == i:
                shutil.move(os.path.join(root, file), new_test)
print("标签分配完毕！\n Success!")
