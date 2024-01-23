import numpy as np

def iou(box,boxes):
    #计算面积：[x1,y1,x2,y2]
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    #找交集
    xx1 = np.maximum(box[0],boxes[:,0])#横坐标，左上角的最大值
    yy1 = np.maximum(box[1],boxes[:,1])#纵坐标，左上角的最大值
    xx2 = np.minimum(box[2],boxes[:,2])#横坐标，右下角的最小值
    yy2 = np.minimum(box[3],boxes[:,3])#纵坐标，右下角的最小值

    #判断是否有交集
    w = np.maximum(0,xx2-xx1)
    h = np.maximum(0,yy2-yy1)

    #交集的面积
    inter = w*h
    ovr = np.true_divide(inter,(box_area+boxes_area-inter))
    return ovr

if __name__ == '__main__':
    a = np.array([1,1,11,11])
    b = np.array([[1,1,10,10],[10,10,20,20],[12,12,34,34]])
    print(iou(a,b))