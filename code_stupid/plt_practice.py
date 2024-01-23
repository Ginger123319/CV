import numpy as np
from PIL import Image
from pathlib import Path
from xml.etree import ElementTree as ET  # xml文件解析方法


# （3）处理超出边缘的检测框
def merge_bboxes(bboxes, cutx, cuty):
    # 保存修改后的检测框
    merge_box = []

    # 遍历每张图像，共4个
    for i, box in enumerate(bboxes):

        # 每张图片中需要删掉的检测框
        index_list = []

        # 遍历每张图的所有检测框,index代表第几个框
        for index, box in enumerate(box):

            # axis=1纵向删除index索引指定的列，axis=0横向删除index指定的行
            # box[0] = np.delete(box[0], index, axis=0)

            # 获取每个检测框的宽高
            x1, y1, x2, y2 = box

            # 如果是左上图，修正右侧和下侧框线
            if i == 0:
                # 如果检测框左上坐标点不在第一部分中，就忽略它
                if x1 > cutx or y1 > cuty:
                    index_list.append(index)

                # 如果检测框右下坐标点不在第一部分中，右下坐标变成边缘点
                if y2 >= cuty >= y1:
                    y2 = cuty
                    if y2 - y1 < 5:
                        index_list.append(index)

                if x2 >= cutx >= x1:
                    x2 = cutx
                    # 如果修正后的左上坐标和右下坐标之间的距离过小，就忽略这个框
                    if x2 - x1 < 5:
                        index_list.append(index)

            # 如果是右上图，修正左侧和下侧框线
            if i == 1:
                if x2 < cutx or y1 > cuty:
                    index_list.append(index)

                if y2 >= cuty >= y1:
                    y2 = cuty
                    if y2 - y1 < 5:
                        index_list.append(index)

                if x1 <= cutx <= x2:
                    x1 = cutx
                    if x2 - x1 < 5:
                        index_list.append(index)

            # 如果是左下图
            if i == 2:
                if x1 > cutx or y2 < cuty:
                    index_list.append(index)

                if y1 <= cuty <= y2:
                    y1 = cuty
                    if y2 - y1 < 5:
                        index_list.append(index)

                if x1 <= cutx <= x2:
                    x2 = cutx
                    if x2 - x1 < 5:
                        index_list.append(index)

            # 如果是右下图
            if i == 3:
                if x2 < cutx or y2 < cuty:
                    index_list.append(index)

                if x1 <= cutx <= x2:
                    x1 = cutx
                    if x2 - x1 < 5:
                        index_list.append(index)

                if y1 <= cuty <= y2:
                    y1 = cuty
                    if y2 - y1 < 5:
                        index_list.append(index)

            # 更新坐标信息
            bboxes[i][index] = [x1, y1, x2, y2]  # 更新第i张图的第index个检测框的坐标

        # 删除一张图片不满足要求的框，并保存
        merge_box.append(np.delete(bboxes[i], index_list, axis=0))

    # 返回坐标信息
    return merge_box


# 对传入的四张图片数据增强
def get_mosaic_data(image_list, input_shape, label_box=True):
    h, w = input_shape  # 获取图像的宽高

    '''设置拼接的分隔线位置'''
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)  # 0.6
    scale_high = scale_low + 0.2  # 0.8

    image_datas = []  # 存放图像信息
    box_datas = []  # 存放检测框信息
    index = 0  # 当前是第几张图

    # (1)图像分割,处理每一张图片
    for frame_list in image_list:

        frame = frame_list[0]  # 取出的某一张图像
        box = np.array(frame_list[1])  # 该图像对应的检测框坐标,shape(1,n,4)

        ih, iw = frame.shape[0:2]  # 图片的宽高
        if label_box:
            cx = (box[:, 0] + box[:, 2]) // 2  # 检测框中心点的x坐标
            cy = (box[:, 1] + box[:, 3]) // 2  # 检测框中心点的y坐标

        # 对输入图像缩放
        new_ar = w / h  # 图像的宽高比
        scale = np.random.uniform(scale_low, scale_high)  # 缩放0.6--0.8倍
        # 图片调整后的宽高
        nh = int(scale * h)  # 缩放比例乘以要求的宽高
        nw = int(nh * new_ar)  # 保持原始宽高比例

        # 缩放图像
        frame = cv2.resize(frame, (nw, nh))

        if label_box:
            # 调整中心点坐标
            cx = cx * nw / iw
            cy = cy * nh / ih

            # 调整检测框的宽高
            bw = (box[:, 2] - box[:, 0]) * nw / iw  # 修改后的检测框的宽高
            bh = (box[:, 3] - box[:, 1]) * nh / ih

        # 创建一块[416,416]的底版
        new_frame = np.zeros((h, w, 3), np.uint8)

        # 确定每张图的位置
        if index == 0:
            new_frame[0:nh, 0:nw] = frame  # 第一张位于左上方
        elif index == 1:
            new_frame[0:nh, w - nw:w] = frame  # 第二张位于右上方
        elif index == 2:
            new_frame[h - nh:h, 0:nw] = frame  # 第三张位于左下方
        elif index == 3:
            new_frame[h - nh:h, w - nw:w] = frame  # 第四张位于右下方

        if label_box:
            # 修正每个检测框的位置（由于标签的位置需要和底版的坐标系对应）
            if index == 0:  # 左上图像
                box[:, 0] = cx - bw // 2  # x1
                box[:, 1] = cy - bh // 2  # y1
                box[:, 2] = cx + bw // 2  # x2
                box[:, 3] = cy + bh // 2  # y2

            if index == 1:  # 右上图像
                box[:, 0] = cx - bw // 2 + w - nw  # x1
                box[:, 1] = cy - bh // 2  # y1
                box[:, 2] = cx + bw // 2 + w - nw  # x2
                box[:, 3] = cy + bh // 2  # y2

            if index == 2:  # 左下图像
                box[:, 0] = cx - bw // 2  # x1
                box[:, 1] = cy - bh // 2 + h - nh  # y1
                box[:, 2] = cx + bw // 2  # x2
                box[:, 3] = cy + bh // 2 + h - nh  # y2

            if index == 3:  # 右下图像
                box[:, 0] = cx - bw // 2 + w - nw  # x1
                box[:, 1] = cy - bh // 2 + h - nh  # y1
                box[:, 2] = cx + bw // 2 + w - nw  # x2
                box[:, 3] = cy + bh // 2 + h - nh  # y2

        index = index + 1  # 处理下一张

        # 保存处理后的图像及对应的检测框坐标
        image_datas.append(new_frame)
        box_datas.append(box)

    # （2）将四张图像拼接在一起
    # 在指定范围中选择横纵向分割线
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    # 创建一块[416,416]的底版用来组合四张图
    # 实际上就是将四块同样大小的图片，分别都切四份；然后各自取对应的那一份拼起来
    # 比如左上角就取左上角有图片信息的那张图片，右上角就选右上角有图片信息的图片
    new_image = np.zeros((h, w, 3), np.uint8)
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[:cuty, cutx:, :] = image_datas[1][:cuty, cutx:, :]
    new_image[cuty:, :cutx, :] = image_datas[2][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[3][cuty:, cutx:, :]

    # 显示合并后的图像
    # plt.imshow(new_image)
    # plt.show()
    if label_box:
        # 处理超出图像边缘的检测框
        box_datas = merge_bboxes(box_datas, cutx, cuty)
        box_datas = np.concatenate(box_datas, axis=0)
    return new_image, box_datas


def plot_results(pii_img, pii_name, save_path):
    from matplotlib import pyplot as plt
    # 创建一个新的图形窗口
    plt.figure(figsize=(8, 4))
    plt.imshow(pii_img)

    # ax = plt.gca()

    plt.axis('off')
    # 保存图形到指定路径,必须要在show之前调用，否则只会保存一个空的图片

    plt.savefig(Path(save_path, pii_name))
    plt.show()


if __name__ == '__main__':
    img_path = r'D:\Python\test_jzyj\key_mouse_20\pic\0001.jpg'
    img = Image.open(img_path)
    print(img.size)
    img_name = img_path.split('\\')[-1]
    plot_results(img, img_name, 'output')
