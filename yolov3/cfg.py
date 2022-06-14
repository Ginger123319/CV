param_path = r"..\..\source\YOLO\param.pt"
IMG_HEIGHT = 416
IMG_WIDTH = 416

# CLASS_NUM = 10
# 不做one-hot，所以只需要一个值作为标签即可
CLASS_NUM = 1
# 注意此处是以宽高形式存放的
ANCHORS_GROUP = {
    13: [[69, 301],
         [103, 325],
         [107, 344]],
    26: [[48, 94],
         [58, 128],
         [63, 130]],
    52: [[29, 36],
         [39, 78],
         [43, 90]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
