import os
import shutil

saveBasePath = "./VOC2007/ImageSets/Main/"
# 运行 04.标签分包.py 代码，将 VOC2007/labels 路径下的标签，按照上面分好的数据集，
# 分别保存到 labels/test、labels/train、labels/val
with open(os.path.join(saveBasePath, 'train.txt')) as f:
    train_number = f.readlines()

with open(os.path.join(saveBasePath, 'val.txt')) as f:
    val_number = f.readlines()

with open(os.path.join(saveBasePath, 'test.txt')) as f:
    test_number = f.readlines()

train_image = []
val_image = []
test_image = []

for i in train_number:
    i = i.rstrip('\n')
    i_image = i + '.txt'
    # print(i_image)
    train_image.append(i_image)
print(train_image)

for i in val_number:
    i = i.rstrip('\n')
    i_image = i + '.txt'
    # print(i_image)
    val_image.append(i_image)
print(val_image)

for i in test_number:
    i = i.rstrip('\n')
    i_image = i + '.txt'
    # print(i_image)
    test_image.append(i_image)
print(test_image)

new_train = "./labels/train/"
if not os.path.exists('./labels/train/'):
    os.makedirs('./labels/train/')
new_val = "./labels/val/"
if not os.path.exists('./labels/val/'):
    os.makedirs('./labels/val/')
new_test = "./labels/test/"
if not os.path.exists('./labels/test/'):
    os.makedirs('./labels/test/')

dir_path = "./VOC2007/labels/"
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
