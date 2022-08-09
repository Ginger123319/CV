import numpy as np


# 重叠率
def iou(box, boxes, isMin=False):  # 1st框，一堆框
    # 计算面积：box = [x1,y1,x2,y2] boxes = [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    # 张量乘法，对应位置相乘，计算出各个矩形框的面积大小，用一个矢量装起来
    # boxes[:, 2]这就是一个矢量
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 找交集的左上角和右下角坐标
    # 实际上还是两个两个框进行求IOU，只是通过张量的运算将重复步骤合并了
    # 假设框之间有交集：那么左上角的坐标为两个框中左上角坐标的最大值
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    # 交集右上角坐标为两个框中的最小值
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 判断是否有交集:有交集，交集的右下角坐标一定大于左上角坐标
    # 因此在求交集矩形的宽高时与0作比较就能判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # 交集的面积
    inter = w * h  # 对应位置元素相乘
    # isMin(IOU有两种：一个除以最小值，一个除以并集)
    if isMin:
        # 最小面积的IOU：O网络用.避免出现大框套小框的情况
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  # 并集的IOU：P和R网络用；交集/并集

    return ovr


# 非极大值抑制[借助iou计算来淘汰框]
# 思路：首先根据对置信度进行排序，找出最大值框与剩下的每个框做IOU比较
# 再将保留下来的框再进行循环比较，每次保留置信度最大的框
def nms(boxes, thresh=0.5, isMin=False):
    # 当输入的数组中没有元素时返回空数组(防止程序有缺陷报错)
    # print(boxes.shape)
    if boxes.shape[0] == 0:
        return np.array([])

    # 根据置信度排序：[x1,y1,x2,y2,C]
    # 有很多框,不止一个,首先进行切片获取到置信度,拿到排序后的索引,再用索引去取原张量的元素赋给新张量
    # np.argsort(boxes[:, 4])返回的是从小到大排序的索引
    # np.argsort(-boxes[:, 4])返回的是从大到小排序的索引
    _boxes = boxes[np.argsort(-boxes[:, 4])]
    # print(boxes)
    # 创建空列表，存放保留剩余的框
    r_boxes = []
    # 用1st个框，与其余的框进行比较，当长度等于1时停止
    while _boxes.shape[0] > 1:  # shape[0]代表0轴上框的个数
        # 取出第1个框
        a_box = _boxes[0]
        # 取出剩余的框
        b_boxes = _boxes[1:]

        # 将1st个框加入列表
        r_boxes.append(a_box)  # 每循环一次往，添加一个置信度最高的框
        # print(iou(a_box, b_boxes))

        # 比较IOU，将符合阈值条件的的框保留下来
        # iou(a_box, b_boxes)返回的是一个矢量,表示第一个框和各个框的IOU大小
        # 将阈值小于thresh的建议框保留下来，返回保留框的索引
        index = np.where(iou(a_box, b_boxes) < thresh)
        print(index)
        # 在剩下的框中选择需要保留框加入到下一次比较的数组中
        _boxes = b_boxes[index]
        # print(_boxes)

    if _boxes.shape[0] > 0:  # 最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0])  # 将此框添加到列表中
    # stack组装为矩阵：:将列表中的数据在0轴上堆叠（行方向）
    return np.stack(r_boxes)


# 扩充：找到左上角坐标，及最大边长，沿着最小边长的两边扩充
def convert_to_square(bbox):  # 将矩形框，补齐转成正方形框
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]  # 框高
    w = bbox[:, 2] - bbox[:, 0]  # 框宽
    max_side = np.maximum(h, w)  # 返回最大边长
    # 求新的左上角坐标，有可能超出原图
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    # 求新的右下角坐标
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == '__main__':
    # a = np.array([1, 1, 11, 11])
    # bs = np.array([[1, 1, 10, 10], [11, 11, 20, 20]])
    # # 使用isMin，iou会大一些
    # print(iou(a, bs, True))

    bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
    # bs = np.array([])
    # print(bs[:,3].argsort())
    print(nms(bs))

# ★★★生成样本是不注释会有影响
# #注释：
# #例子1--max与maxium的区别
# import numpy as np
#
# a = [-2,-1,0,1,2]
# print(np.max(a)) #接收一个参数，返回最大值
# print(np.maximum(0,a)) #接收两个参数，X与Y逐个比较取其最大值：若比0小返回0，若比0大返回较大值
#
# #例子2--true_divide等价于divide,在python3中
#
# #例子3----argsort根据索引排序
# b = np.array([[5,3],[3,2],[1,6]])
# index = -b[:,1].argsort() #列出b[:,1]对应元素的负值索引
# print(b[index]) #根据b[:,1]对应元素的索引由大到小排序，不加符号由小到大排序；[[1 6]，[5 3],[3 2]]
#
# #例子4----a[np.where(a<3)]:返回符合条件的值
# c = np.array([5,3,2,1,6])
#
# print(c<3) #[False False  True  True False]
# index = np.where(c<3) #返回符合条件的索引
# print(index) #[2, 3]
# print(a[index]) #返回符合条件的值：[2 1]
#
# #例子5----np.stack的用方法
# a = np.array([1,2])
# b = np.array([3,4])
# c = np.array([5,6])
#
# list = []
#
# list.append(a)
# list.append(b)
# list.append(c)
#
# print(list) # [array([1, 2]), array([3, 4]), array([5, 6])]
#
# d = np.stack(list)
# print(d) #[[1 2][3 4][5 6]]
