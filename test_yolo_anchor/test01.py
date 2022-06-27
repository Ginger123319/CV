import glob
import xml.etree.ElementTree as ET

import numpy as np
import torch

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = r"D:\Python\source\key_mouse\outputs"
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        try:
            for obj in tree.iter("object"):
                # 此处做归一化的目的是将聚类时的数据缩小，减少运算量，提升计算速度
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bndbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)
                dataset.append([xmax - xmin, ymax - ymin])
        except:
            print(xml_file)
    return np.array(dataset)


if __name__ == '__main__':
    # print(__file__)
    data = load_dataset(ANNOTATIONS_PATH)
    print(data)
    print(data.shape)
    out = kmeans(data, k=CLUSTERS)
    # clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    # out= np.array(clusters)/416.0
    print(out)
    # 计算建议框和数据框的IOU，测试建议框选取的好坏程度
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    # 网络设置的输入尺寸是416*416，所有图片都必须等比缩放为416*416，由于等比缩放不改变比例值
    # 所以计算缩放后建议框的w和h就乘以416即可
    print("Boxes:\n {}-{}".format(np.ceil(out[:, 0] * 416), np.ceil(out[:, 1] * 416)))
    w_ndarray = np.sort(np.ceil(out[:, 0] * 416)).astype(np.int32)
    h_ndarray = np.sort(np.ceil(out[:, 1] * 416)).astype(np.int32)
    # print(w_ndarray)
    ANCHORS_ndarray = np.stack((w_ndarray, h_ndarray), axis=0)
    # print(torch.tensor(ANCHORS_ndarray.T[0:3]), torch.tensor(ANCHORS_ndarray.T[3:6]),
    #       torch.tensor(ANCHORS_ndarray.T[6:9]), sep="\n")
    print(np.split(ANCHORS_ndarray.T, len(ANCHORS_ndarray.T) // 3))
    # ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    # print("Ratios:\n {}".format(sorted(ratios)))
