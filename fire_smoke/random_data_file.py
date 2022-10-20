import os
import random

random.seed(0)
img_path = "../../source/fire_smoke/data/smoke_20220816/smoke-all-imgs-0415"
label_path = "../../source/yolodata/smoke/labels/train"
saveBasePath = "../../source/yolodata/smoke/main"

trainval_percent = 0.98
train_percent = 0.994

temp_txt = os.listdir(label_path)
total_txt = []
for txt in temp_txt:
    if txt.endswith(".txt"):
        total_txt.append(txt)

num = len(total_txt)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("val size", tv - tr)
print("train size", tr)
print("test size", num - tv)

ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_txt[i][:-4] + '\n'
    if i in trainval:
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrain.close()
fval.close()
ftest.close()
