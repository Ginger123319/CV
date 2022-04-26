IMG_HEIGHT = 416
IMG_WIDTH = 416

# CLASS_NUM = 10
# 不做one-hot，所以只需要一个值作为标签即可
CLASS_NUM = 1
# 注意此处是以宽高形式存放的
ANCHORS_GROUP = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
