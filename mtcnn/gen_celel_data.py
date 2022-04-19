import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

# 测试查看：生成样本（以5张图片为例）
# anno_src = r"D:\MTCNN-lihao\Cebela-new\Anno\list_bbox_celeba.txt"
# img_dir = r"D:\MTCNN-lihao\Cebela-new\img_celeba"
# save_path = r"D:\celeba_testing"

# 正式生成样本
# 原始数据样本、标签路径
anno_src = r"D:\Python\source\FACE\celebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\Python\source\FACE\celebA\img\img_celeba\img_celeba"

# 样本保存路径
save_path = r"D:\Python\source\FACE\celebA\save_pic_label"

# 生成不同尺寸的人脸样本，包括人脸（正样本）、非人脸（负样本）、部分人脸
for face_size in [12, 24, 48]:

    print("gen %i image" % face_size)  # %i:十进制数占位符
    # “样本图片”存储路径--image
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")  # 三级文件路径
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):  # 如果文件不存在则创建文件路径
            os.makedirs(dir_path)

    # “样本标签”存储路径--text
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")  # 创建正样本txt文件
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    # 计数初始值:给文件命名
    if face_size == 12:
        positive_count = 37223  # 计数器初始值
        negative_count = 137445
        part_count = 63418
    elif face_size == 24:
        positive_count = 36981  # 计数器初始值
        negative_count = 126232
        part_count = 63675
    else:
        positive_count = 37103  # 计数器初始值
        negative_count = 109632
        part_count = 63543

    # 凡是文件操作，最好try一下，防止程序出错奔溃
    try:
        positive_anno_file = open(positive_anno_filename, "a")  # 以写入的模式打开txt文档
        negative_anno_file = open(negative_anno_filename, "a")
        part_anno_file = open(part_anno_filename, "a")

        for i, line in enumerate(open(anno_src)):  # 枚举出所有信息
            if i < 20002:
                continue  # i小于2时继续读文件readlines
            try:
                print(line)
                strs = line.strip().split()

                image_filename = strs[0].strip()

                image_file = os.path.join(img_dir, image_filename)  # 创建文件绝对路径

                with Image.open(image_file) as img:  # 打开图片文件
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())  # 取2nd个值去除两边的空格，再转为float型
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    px1 = 0  # float(strs[5].strip())  # 人的五官
                    py1 = 0  # float(strs[6].strip())
                    px2 = 0  # float(strs[7].strip())
                    py2 = 0  # float(strs[8].strip())
                    px3 = 0  # float(strs[9].strip())
                    py3 = 0  # float(strs[10].strip())
                    px4 = 0  # float(strs[11].strip())
                    py4 = 0  # float(strs[12].strip())
                    px5 = 0  # float(strs[13].strip())
                    py5 = 0  # float(strs[14].strip())

                    # 过滤字段，去除不符合条件的坐标
                    if max(w, h) < 48 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]  # 左上角和右下角四个坐标点；二维的框有批次概念

                    # 计算出人脸中心点位置：框的中心位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍以图片中心点随机偏移
                    for _ in range(5):  # 每个循环5次，画五个框框、抠出来
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形（12*12，24*24,48*48），并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(
                            1.25 * max(w, h)))  # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8
                        x1_ = np.max(cx_ - side_len / 2, 0)  # 坐标点随机偏移
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 偏移后的新框

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len  # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        offset_px1 = 0  # (px1 - x1_) / side_len   # 人的五官特征的偏移值
                        offset_py1 = 0  # (py1 - y1_) / side_len
                        offset_px2 = 0  # (px2 - x1_) / side_len
                        offset_py2 = 0  # (py2 - y1_) / side_len
                        offset_px3 = 0  # (px3 - x1_) / side_len
                        offset_py3 = 0  # (py3 - y1_) / side_len
                        offset_px4 = 0  # (px4 - x1_) / side_len
                        offset_py4 = 0  # (py4 - y1_) / side_len
                        offset_px5 = 0  # (px5 - x1_) / side_len
                        offset_py5 = 0  # (py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)  # “抠图”，crop剪下框出的图像
                        face_resize = face_crop.resize((face_size, face_size),
                                                       Image.ANTIALIAS)  # ★按照人脸尺寸（“像素矩阵大小”）进行缩放：12/24/48；坐标没放缩

                        iou = utils.iou(crop_box, np.array(boxes))[0]  # 抠出来的框和原来的框计算IOU
                        if iou > 0.6:  # 正样本；原为0.65
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()  # flush：将缓存区的数据写入文件
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))  # 保存
                            positive_count += 1
                        elif iou > 0.3:  # 部分样本；原为0.4
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))  # 写入txt文件
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.01:  # ★这样生成的负样本很少；原为0.3
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        # 生成负样本
                        _boxes = np.array(boxes)

                    for _ in range(15):  # 数量一般和前面保持一样
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.01:  # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿

                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
            except Exception as e:
                traceback.print_exc()  # 如果出现异常，把异常打印出来
            if i > 25002:
                break

    # 关闭写入文件
    finally:
        positive_anno_file.close()  # 关闭正样本txt件
        negative_anno_file.close()
        part_anno_file.close()
