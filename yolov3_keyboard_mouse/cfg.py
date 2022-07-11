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
    13: [[157, 123],
         [223, 160],
         [323, 276]],
    26: [[59, 49],
         [77, 70],
         [97, 98]],
    52: [[20, 11],
         [33, 24],
         [55, 35]]
}
ANCHORS_GROUP1 = {
    0: [[157, 123],
        [223, 160],
        [323, 276]],
    1: [[59, 49],
        [77, 70],
        [97, 98]],
    2: [[20, 11],
        [33, 24],
        [55, 35]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
