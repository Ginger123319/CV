param_path = r"..\..\source\key_mouse\param.pt"
train_path = "data/label_train.txt"
val_path = "data/label_val.txt"
img_path = r"..\..\source\key_mouse"
IMG_HEIGHT = 128
IMG_WIDTH = 128

# CLASS_NUM = 10
# 不做one-hot，所以只需要一个值作为标签即可
CLASS_NUM = 1
# 注意此处是以宽高形式存放的
ANCHORS_GROUP = {
    4: [[79, 57],
        [85, 59],
        [111, 95]],
    8: [[25, 25],
        [45, 34],
        [49, 36]],
    16: [[7, 4],
         [13, 10],
         [23, 16]]
}
ANCHORS_GROUP_AREA = {
    4: [x * y for x, y in ANCHORS_GROUP[4]],
    8: [x * y for x, y in ANCHORS_GROUP[8]],
    16: [x * y for x, y in ANCHORS_GROUP[16]],
}
