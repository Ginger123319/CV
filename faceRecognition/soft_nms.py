import numpy as np


# 单个矩形box:[x1,y1,x2,y2]和多个矩形boxes:[[x1,y1,x2,y2],[x1,y1,x2,y2]]框计算IOU
# is_min:判断IOU的计算方式
# True：交集面积/矩形框中的最小面积 False：交集面积/并集面积（总面积-交集面积）
def iou(box, boxes, is_min=False):
    # 计算面积：[x1,y1,x2,y2]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    # boxes[:, 2]就是一个矢量
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 找交集
    xx1 = np.maximum(box[0], boxes[:, 0])  # 横坐标，左上角的最大值
    yy1 = np.maximum(box[1], boxes[:, 1])  # 纵坐标，左上角的最大值
    xx2 = np.minimum(box[2], boxes[:, 2])  # 横坐标，右下角的最小值
    yy2 = np.minimum(box[3], boxes[:, 3])  # 纵坐标，右下角的最小值

    # 判断是否有交集，有就是后面的值作为宽高，没有就是0
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 交集的面积
    inter = w * h

    if is_min:
        ovr = np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        ovr = np.true_divide(inter, (box_area + boxes_area - inter))
    return ovr


# 会导致保留框的置信度发生变化，会产生什么影响？
# soft为了保留两个贴在一起的人脸框，根据传统做法，总会有一个框被删除；
def soft_nms(boxes, thresh=0.3, is_min=False, is_soft_nms=False):
    if boxes.shape[0] == 0:
        return np.array([])
    # 根据置信度排序：[x1,y1,x2,y2,c]
    _boxes = boxes[np.argsort((-boxes[:, 4]))]

    # 创建空列表，存放保留的框
    r_boxes = []
    while _boxes.shape[0] > 1:
        # 取出第一个框
        a_box = _boxes[0]
        # 取出剩余的框（候选框）
        b_boxes = _boxes[1:]
        # 将剩余框的置信度都保留下来
        score = b_boxes[:, 4]
        # 将1st个框保留
        r_boxes.append(a_box)
        if is_soft_nms:
            score_thresh = 0.02
            index = np.where(iou(a_box, b_boxes, is_min) > thresh)
            # 将IOU大于阈值的框的置信度都进行更新
            # 此处iou(a_box, b_boxes, is_min)是一个矢量
            # 需要取出对应索引的iou进行计算
            score[index] *= (1 - iou(a_box, b_boxes, is_min))[index]
            # 将置信度小于置信度阈值的框从候选框中删除
            index = np.where(score < score_thresh)
            print(score)
            _boxes = np.delete(b_boxes, index, axis=0)
        else:

            # 比较IOU，保留IOU小于阈值的框
            # print(iou(a_box, b_boxes, is_min)) 是一个矢量
            index = np.where(iou(a_box, b_boxes, is_min) < thresh)
            _boxes = b_boxes[index]
    # 保留最后一个框shape[0]==1
    # 因为最后一个框和上一个保留下来的框极大可能不是同一个目标
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    # 将保留框的列表中的np数组堆叠在一起，返回一个np数组，增加一个维度对框计数
    return np.stack(r_boxes)


if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # b = np.array([[1,1,10,10],[10,10,20,20],[12,12,34,34]])
    # print(iou(a,b))
    bs = np.array([[1, 1, 10, 10, 0.4], [1, 1, 9, 9, 0.1], [9, 8, 13, 20, 0.15], [6, 1, 18, 17, 0.13]])
    print(soft_nms(bs, thresh=0.1, is_min=False, is_soft_nms=True))
