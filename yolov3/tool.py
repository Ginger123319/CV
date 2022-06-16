import numpy as np
import torch


def ious(box, boxes, isMin=False):  # 定义iou函数
    box_area = (box[3] - box[1]) * (box[4] - box[2])  # 计算置信度最大框的面积
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])  # 计算其他所有框的面积
    xx1 = torch.max(box[1], boxes[:, 1])  # 计算交集左上角x的坐标其他同理
    yy1 = torch.max(box[2], boxes[:, 2])
    xx2 = torch.min(box[3], boxes[:, 3])
    yy2 = torch.min(box[4], boxes[:, 4])

    # w = torch.max(0, xx2 - xx1)
    # h = torch.max(0, yy2 - yy1)#获取最大值也可以用下面这种方法
    w = torch.clamp(xx2 - xx1, min=0)  # 获取最大值
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h  # 计算交集面积

    if isMin:  # 用于判断是交集/并集，还是交集/最小面积（用于去掉同一个目标大框套小框中的小框）

        ovr = inter / torch.min(box_area, area)
    else:
        ovr = inter / (box_area + area - inter)
    print("ovr is:", ovr)
    return ovr


def nms(boxes, thresh=0.001, isMin=True):  # 定义nms函数并传3个参数，分别是框，置信度阀值，是否最小面积

    if boxes.shape[0] == 0:  # 获取框的个是看是否为0，为0没框就返回一个空的数组防止代码报错
        return np.array([])

    _boxes = boxes[(-boxes[:, 0]).argsort()]  # 对框进行排序按置信度从大到小的顺序
    r_boxes = []  # 定义一个空的列表用来装合格的框

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]  # 取出第一个（置信度最大的框）框作为目标框与其他框做iou
        b_boxes = _boxes[1:]  # 取出剩下的所有框

        r_boxes.append(a_box)  # 将第一个框添加到列表

        index = np.where(ious(a_box, b_boxes, isMin) < thresh)  # 对框做iou将满足iou阀值条件的框留下并返回其索引
        _boxes = b_boxes[index]  # 根据索引取框并赋值给_boxes，使其覆盖原来的_boxes
    if _boxes.shape[0] > 0:  # 判断是否剩下最后一个框
        r_boxes.append(_boxes[0])  # 将最后一个框，说明这是不同物体，并将其放进列表

    return torch.stack(r_boxes)


if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[11,11,20,20]])
    # print(iou(a,bs))

    bs = torch.tensor([[1, 1, 10, 10, 12, 8], [5, 1, 11, 5, 15, 9], [9, 8, 13, 9, 15, 3], [6, 11, 18, 17, 40, 2]])
    # print(bs[:,3].argsort())
    print(nms(bs))
