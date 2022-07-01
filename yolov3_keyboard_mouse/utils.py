import torch
import cv2 as cv

target_len = 128


def pic_resize(pic_path):
    img_data = cv.imread(pic_path)
    # cv.imshow("pic", img_data)
    # cv.waitKey()
    # cv.destroyAllWindows()
    h = img_data.shape[0]
    w = img_data.shape[1]
    max_len = max(h, w)
    # 计算缩放比例
    ratio = target_len / max_len
    h = int(h * ratio)
    w = int(w * ratio)
    # print(h, w)
    # 按比例缩放
    dst = cv.resize(img_data, (w, h))
    # 给缩放后的图片加黑边，在下面或者右边添加
    # 不需要分类，有一条边为128，计算结果就为0
    dst = cv.copyMakeBorder(dst, 0, 128 - h, 0, 128 - w, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # print(dst.shape)
    # cv.imshow("pic", dst)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # 返回处理后的图片数据以及缩放比例
    return dst, ratio


def iou(box, boxes, mode="inter"):
    box_area = (box[3] - box[1]) * (box[4] - box[2])
    boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

    x1 = torch.max(box[1], boxes[:, 1])
    y1 = torch.max(box[2], boxes[:, 2])
    x2 = torch.min(box[3], boxes[:, 3])
    y2 = torch.min(box[4], boxes[:, 4])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter)
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)


def nms(boxes, thresh, mode='inter'):
    args = boxes[:, 0].argsort(descending=True)
    sort_boxes = boxes[args]
    keep_boxes = []

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            _iou = iou(_box, _boxes, mode)
            sort_boxes = _boxes[_iou < thresh]
        else:
            break

    return keep_boxes


# def detect(feature_map, thresh):
#     masks = feature_map[:, 4, :, :] > thresh
#     idxs = torch.nonzero(masks)


if __name__ == '__main__':
    box = torch.Tensor([2, 2, 3, 3, 6])
    boxes = torch.Tensor([[2, 2, 3, 3, 6], [2, 2, 4, 4, 5], [2, 2, 5, 5, 4]])
    print(iou(box, boxes, mode="inter"))
    print(nms(boxes, 0.1))
    # import numpy as np
    #
    # a = np.array([[1, 2], [3, 4]])
    # print(a[:, 1])
