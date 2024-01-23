import os
import random

random.seed(0)

# xmlfilepath = r'./VOC2007/Annotations'
xmlfilepath = r'D:\Python\source\key_mouse\outputs'
saveBasePath = r"./VOC2007/ImageSets/Main/"

# ----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   train_percent不需要修改
# 运行 01.划分数据集.py代码，将数据集分割成 训练集、验证集和测试集（自行设置比例）
# 生成的文件在 VOC2007/ImageSets/Main 中，
# 分别是 test.txt、train.txt、val.txt
# ----------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("traub suze", tr)
# ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        # ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

    # ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
