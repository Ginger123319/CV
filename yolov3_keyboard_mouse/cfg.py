param_path = r"..\..\source\key_mouse\param.pt"
train_path = "data/label_train.txt"
val_path = "data/label_val.txt"
img_path = r"..\..\source\key_mouse"
IMG_HEIGHT = 416
IMG_WIDTH = 416

# CLASS_NUM = 10
# 不做one-hot，所以只需要一个值作为标签即可
CLASS_NUM = 1
# 注意此处是以宽高形式存放的
ANCHORS_GROUP = {
    13: [[167, 154],
         [285, 233],
         [353, 258]],
    26: [[78, 91],
         [131, 104],
         [162, 142]],
    52: [[30, 19],
         [64, 43],
         [66, 61]]
}
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
