import math
import random
import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
from mosaic_all import get_mosaic_data
import torch


# 生成随机数
def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


# 修正一切经过仿射变换的标签
def transform_affine_labels(boxes, M, img):
    height, width = img.shape[:2]
    n = len(boxes)
    xy = np.ones((n * 4, 3))
    # 取出原来的框的四个坐标点
    xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ M.T  # 转换四个坐标点为仿射变换后的坐标
    xy = xy[:, :2].reshape(n, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1), boxes[:, -1])).reshape(5, n).T
    # clip
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    return new.astype(np.uint)


def iou(ori_bbox, bbox, mode):
    if mode == "cut":
        ori_bbox_area = (ori_bbox[:, 2] - ori_bbox[:, 0]) * (ori_bbox[:, 3] - ori_bbox[:, 1])
        bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        ori_bbox_area = np.clip(ori_bbox_area, 1, a_max=None)
        return bbox_area / ori_bbox_area
    elif mode == "wipe":
        ori_bbox_area = (ori_bbox[:, 2] - ori_bbox[:, 0]) * (ori_bbox[:, 3] - ori_bbox[:, 1])
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        x1 = torch.max(bbox[0], ori_bbox[:, 0])
        y1 = torch.max(bbox[1], ori_bbox[:, 1])
        x2 = torch.min(bbox[2], ori_bbox[:, 2])
        y2 = torch.min(bbox[3], ori_bbox[:, 3])

        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)

        inter = w * h
        return inter / ori_bbox_area
    else:
        raise Exception("暂不支持此种方式！")


# 随机进行上下左右翻转
def flip(image_label, label_type, augment_scope=None):
    img = image_label[0]
    target = image_label[1]
    h, w = img.shape[:2]
    flag = random.randint(-1, 1)

    # 目标检测
    if label_type == "103" and len(target) != 0:

        target = np.array(target)
        if augment_scope == 'all':
            img = cv2.flip(img, flag)
            # 上下左右翻转
            if -1 == flag:
                target[:, [0, 2]] = w - target[:, [2, 0]]
                target[:, [1, 3]] = h - target[:, [3, 1]]
            # 上下翻转
            elif 0 == flag:
                target[:, [1, 3]] = h - target[:, [3, 1]]
            # 左右翻转
            elif 1 == flag:
                target[:, [0, 2]] = w - target[:, [2, 0]]
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                img[ymin:ymax, xmin:xmax] = cv2.flip(img[ymin:ymax, xmin:xmax], flag)
        return img, target.tolist()
    else:
        img = cv2.flip(img, flag)
        return img, target


# 按照给定的左上角和右下角坐标进行裁剪
def img_cut(img):
    """
    :assert 0.1 < lu_max <= 0.2
    :assert 0.8 <= rb_min < 1
    """
    lu_max = 0.2
    rb_min = 0.8
    crop_x1y1 = np.random.uniform(0.1, lu_max, 2)
    crop_x2y2 = np.random.uniform(rb_min, 1, 2)
    x1, y1 = crop_x1y1
    x2, y2 = crop_x2y2

    h, w = img.shape[:2]

    crop_x1 = int(x1 * w)
    crop_y1 = int(y1 * h)
    crop_x2 = int(x2 * w)
    crop_y2 = int(y2 * h)

    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img, crop_x1, crop_y1, crop_x2, crop_y2


# 按照给定的左上角和右下角坐标进行裁剪
def cut(image_label, label_type, augment_scope=None):
    thresh = 0.3
    img = image_label[0]
    target = image_label[1]

    # 目标检测
    if label_type == "103" and len(target) != 0:
        target = np.array(target)
        ori_bbox = np.copy(target)

        if augment_scope == "all":
            crop_img, crop_x1, crop_y1, crop_x2, crop_y2 = img_cut(img)
            # 先做坐标转换
            target[:, [0, 2]] = target[:, [0, 2]] - crop_x1
            target[:, [1, 3]] = target[:, [1, 3]] - crop_y1
            target[:, [1, 3]] = np.clip(target[:, [1, 3]], 0, crop_y2 - crop_y1)
            target[:, [0, 2]] = np.clip(target[:, [0, 2]], 0, crop_x2 - crop_x1)

            # 对于裁剪后与原始标签框的交并比小于等于阈值的框直接丢弃
            _iou = iou(ori_bbox, target, mode="cut")
            target = target[_iou > thresh]
            return crop_img, target.tolist()
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                w, h = ((xmax - xmin), (ymax - ymin))
                crop_img = img_cut(img[ymin:ymax, xmin:xmax])[0]
                img[ymin:ymax, xmin:xmax] = cv2.resize(crop_img, (w, h))

            return img, target.tolist()
    else:
        crop_img = img_cut(img)[0]
        return crop_img, target


def change_color(img, rgain=0.5, ggain=0.5, bgain=0.5):
    if rgain or ggain or bgain:
        # 随机取-1到1三个实数，乘以 rgb 三通道扰动系数
        # r:[1-gain,1+gain]
        r = np.random.uniform(-1, 1, 3) * [rgain, ggain, bgain] + 1  # random gains

        # cv2.split：通道拆分
        # r:[0~255], g:[0~255], b:[0~255]
        red, green, blue = cv2.split(img)
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_red = np.clip(x * r[0], 0, 255).astype(dtype)
        lut_green = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_blue = np.clip(x * r[2], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        # 拿通道上的像素值作为索引到查找表中查找对应的值来替换
        # 大概率拿到近似的值，通过扰动系数来进行控制
        red = cv2.LUT(red, lut_red)
        green = cv2.LUT(green, lut_green)
        blue = cv2.LUT(blue, lut_blue)

        # 通道合并
        img_rgb = cv2.merge((red, green, blue)).astype(dtype)
        return img_rgb


# 随机改变颜色
def color(image_label, label_type, augment_scope=None):
    # rgb 三通道扰动系数
    rgain = 0.5
    ggain = 0.5
    bgain = 0.5

    img = image_label[0]
    target = image_label[1]
    # 目标检测
    if label_type == "103" and len(target) != 0:

        if augment_scope == 'all':
            img = change_color(img, rgain, ggain, ggain)
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                img[ymin:ymax, xmin:xmax] = change_color(img[ymin:ymax, xmin:xmax], rgain, ggain, ggain)
        return img, target
    else:
        img = change_color(img, rgain, ggain, ggain)
        return img, target


def change_brightness(img, vgain):
    if vgain:
        # r:[1+0.5gain,1+gain](变得更亮并且更加饱和)
        r = np.random.uniform(0.5, 1, 1) * [vgain] + 1  # random gains
        image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # cv2.split：通道拆分
        # h:[0~180]色彩, s:[0~255]饱和度, v:[0~255]亮度
        hue, sat, val = cv2.split(image_hsv)
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_val = np.clip(x * r[0], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        val = cv2.LUT(val, lut_val)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)
        # 将hsv格式转为BGR格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image_dst


# 增强亮度
def brightness(image_label, label_type, augment_scope):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param vgain:       HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    vgain = 0.4

    img = image_label[0]
    target = image_label[1]

    # 目标检测
    if label_type == "103" and len(target) != 0:

        if augment_scope == 'all':
            img = change_brightness(img, vgain)
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                img[ymin:ymax, xmin:xmax] = change_brightness(img[ymin:ymax, xmin:xmax], vgain)
        return img, target
    else:
        img = change_brightness(img, vgain)
        return img, target


def change_saturation(img, sgain):
    if sgain:
        # r:[1+0.5gain,1+gain](变得更亮并且更加饱和)
        r = np.random.uniform(0.5, 1, 1) * [sgain] + 1  # random gains
        image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # cv2.split：通道拆分
        # h:[0~180]色彩, s:[0~255]饱和度, v:[0~255]亮度
        hue, sat, val = cv2.split(image_hsv)
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_sat = np.clip(x * r[0], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        sat = cv2.LUT(sat, lut_sat)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)

        # 将hsv格式转为RGB格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image_dst


# 增强饱和度
def saturation(image_label, label_type, augment_scope=None):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param sgain:       HSV 中的 s 扰动系数，yolov5：0.7
    :return:
    """
    sgain = 0.7

    img = image_label[0]
    target = image_label[1]

    # 目标检测
    if label_type == "103" and len(target) != 0:

        if augment_scope == 'all':
            img = change_saturation(img, sgain)
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                img[ymin:ymax, xmin:xmax] = change_saturation(img[ymin:ymax, xmin:xmax], sgain)
        return img, target
    else:
        img = change_saturation(img, sgain)
        return img, target


# 高斯噪声
def gaussian_noise(image_label, label_type, augment_scope, mean=0.1, sigma=0.1):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.copy(image_label[0])
    target = np.array(image_label[1])
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output, target


def add_uniform_noise(img, prob=0.05, value=255):
    """
       随机生成一个0~1的mask，作为椒盐噪声
       :param image:图像
       :param prob: 噪声比例
       :param vaule: 噪声值
       :return:
    """
    rows, cols, chn = img.shape
    noise = np.random.uniform(low=0.0, high=1.0, size=(rows, cols)).astype(dtype=np.float32)
    mask = np.zeros(shape=(rows, cols), dtype=np.uint8) + value
    index = noise > prob
    mask = mask * (~index)
    output = img * index[:, :, np.newaxis] + mask[:, :, np.newaxis]
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output


# 椒盐噪声
def noise(image_label, label_type, augment_scope=None):
    """
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param img: 需要加噪的图片以及其标签，形如[[img:np.ndarray],[[..],[...]]]
    :param noise_num: 添加的噪音点比例，【0.02~0.05之间】
    :return: img_noise
    """
    noise_num = 0.02

    img_noise = image_label[0]
    target = image_label[1]

    # 目标检测
    if label_type == "103" and len(target) != 0:

        if augment_scope == 'all':
            img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 255)
            img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 0)
        else:
            for bbox in target:
                xmin, ymin, xmax, ymax, _ = bbox
                img_noise[ymin:ymax, xmin:xmax] = add_uniform_noise(img_noise[ymin:ymax, xmin:xmax], noise_num * 0.5,
                                                                    255)
                img_noise[ymin:ymax, xmin:xmax] = add_uniform_noise(img_noise[ymin:ymax, xmin:xmax], noise_num * 0.5, 0)
        return img_noise, target
    else:
        img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 255)
        img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 0)
        return img_noise, target


def erase(img, s, r):
    h, w = img.shape[:2]
    i = 0
    # 返回擦除的矩形位置坐标，用于判断是否有标注框被覆盖
    bbox = []
    while i < 3000:
        Se = random.uniform(*s) * h * w
        re = random.uniform(*r)

        He = int(round(math.sqrt(Se * re)))
        We = int(round(math.sqrt(Se / re)))

        xe = random.randint(0, w)
        ye = random.randint(0, h)
        i += 1
        if xe + We <= w and ye + He <= h:
            img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))
            bbox = [xe, ye, xe + We, ye + He]
            break
    return img, bbox


# 随机擦除
def wipe(image_label, label_type, augment_scope=None):
    """Random erasing the an rectangle region in Image.
       Args:
           sl: min erasing area region
           sh: max erasing area region
           r1: min aspect ratio range of earsing region
       """
    sl = 0.02
    sh = 0.1
    r1 = 0.3
    thresh = 0.5

    img = image_label[0]
    target = image_label[1]
    s = (sl, sh)
    r = (r1, 1 / r1)

    # 目标检测
    if label_type == "103" and len(target) != 0:
        if augment_scope == 'part':
            for box in target:
                xmin, ymin, xmax, ymax, _ = box
                img[ymin:ymax, xmin:xmax], _ = erase(img[ymin:ymax, xmin:xmax], s, r)
            return img, target
        else:
            ori_bbox = [label[:4] for label in target]
            ori_bbox = torch.tensor(ori_bbox)
            img, bbox = erase(img, s, r)
            bbox = torch.tensor(bbox)
            _iou = iou(ori_bbox, bbox, mode="wipe")
            target = np.array(target)
            target = target[(_iou < thresh).tolist()]
            return img, target.tolist()
    else:
        img, bbox = erase(img, s, r)
        return img, target


# 增强对比度
def contrast(image_label, label_type, augment_scope=None):
    v = 1.5

    img = image_label[0]
    target = image_label[1]
    # 目标检测
    if label_type == "103" and len(target) != 0 and augment_scope == "part":
        for box in target:
            xmin, ymin, xmax, ymax, _ = box
            part_img = Image.fromarray(img[ymin:ymax, xmin:xmax])
            enh_con = ImageEnhance.Contrast(part_img)
            part_img = enh_con.enhance(v)
            img[ymin:ymax, xmin:xmax] = np.array(part_img)
        return img, target
    else:
        img = Image.fromarray(img)
        enh_con = ImageEnhance.Contrast(img)
        img = enh_con.enhance(v)
        img = np.array(img)
        return img, target


# 围绕中心点，任意角度进行旋转
def rotate(image_label, label_type, augment_scope=None):
    """
    :param cx:中心点x坐标
    :param cy:中心点y坐标
    :param degrees: 旋转角度
    :return: 返回图片和标签坐标信息
    """
    degree = 30
    a = random.uniform(-degree, degree)  # 随机旋转角度

    img = image_label[0]
    target = image_label[1]
    h, w = img.shape[:2]

    cy = h // 2
    cx = w // 2
    M = cv2.getRotationMatrix2D((cx, cy), a, 1)

    cos_degree = np.abs(M[0, 0])
    sin_degree = np.abs(M[0, 1])
    # 修改旋转后图片的宽和高，保证不丢失信息
    new_h = int(w * sin_degree + h * cos_degree)
    new_w = int(h * sin_degree + w * cos_degree)

    # 修改旋转矩阵（在新的宽高，中心点改变）
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    # 目标检测
    if label_type == "103" and len(target) != 0:
        if augment_scope == 'part':
            print("部分图片暂不支持旋转！")
            return img, target
        else:
            # 围绕给定的点逆时针旋转\theta度
            rotate_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=0)
            target = np.array(target, dtype=np.float64)
            # 修改target适配新的宽和高

            target[:, [0, 2]] += (new_w - w) / 2
            target[:, [1, 3]] += (new_h - h) / 2

            target = np.array(target, dtype=np.int64)
            # 标签框坐标转换
            target = transform_affine_labels(target, M, rotate_img)

            return rotate_img, target.tolist()

    else:
        rotate_img = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
        return rotate_img, target


# 缩放变形
def scale(image_label, label_type, augment_scope=None):
    scale = 0.5
    is_Equal = random.randint(0, 1)

    input_img = image_label[0]
    target = image_label[1]

    if is_Equal:
        # 定义宽缩放的倍数
        fx = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
        # 定义高缩放的倍数
        fy = fx
    else:
        fx, fy = np.random.uniform(1 - scale, 1 + scale, 2)
    # 定义一个图像缩放矩阵
    M = np.array([[fx, 0, 0], [0, fy, 0]])
    # 获取图片的宽和高
    height, width = input_img.shape[:2]

    # 目标检测
    if label_type == "103" and len(target) != 0:
        if augment_scope == 'part':
            print("部分图片暂不支持缩放！")
            return input_img, target
        else:
            target = np.array(target)
            # 定义缩放后图片的大小
            scale_img = cv2.warpAffine(input_img, M, (int(width * fx), int(height * fy)))
            # 修改标签
            target = transform_affine_labels(target, M, scale_img)
            # for bbox in target:
            #     xmin, ymin, xmax, ymax,_ = bbox
            #     cv2.rectangle(scale_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            return scale_img, target.tolist()

    else:
        scale_img = cv2.warpAffine(input_img, M, (int(width * fx), int(height * fy)))
        return scale_img, target


# 仿射变换
def affine(image_label, label_type, augment_scope=None):
    degrees = 10
    translate = .1
    scale = 0.2
    shear = 10
    border = 0

    img = image_label[0]
    target = image_label[1]

    height = img.shape[0]
    width = img.shape[1]

    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    # 依次进行各种仿射变换
    M = R @ T @ S
    M = M[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # 目标检测
    if label_type == "103" and len(target) != 0:
        if augment_scope == 'part':
            print("部分图片暂不支持仿射变换！")
            return img, target
        else:
            target = np.array(target)
            # 进行标签转换
            target = transform_affine_labels(target, M, affine_img)
            index_list = []
            for i, bbox in enumerate(target):

                if ((bbox[2] - bbox[0]) < 5) or ((bbox[3] - bbox[1]) < 5):
                    index_list.append(i)
                # else:
                #     xmin, ymin, xmax, ymax,_ = bbox
                #     cv2.rectangle(affine_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            target = np.delete(target, index_list, axis=0)
        return affine_img, target.tolist()


    else:
        return affine_img, target


# 混合两张图片
def mix(image_label, label_type, augment_scope):
    im = image_label[0][0]
    # 原图
    im2 = image_label[1][0]
    labels = image_label[0][1]
    labels2 = image_label[1][1]
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    h, w = im2.shape[:2]

    # 目标检测
    if label_type == "103" and len(labels) != 0:
        if augment_scope == 'part':
            print("部分图片暂不支持混合！")
            return im2, labels2
        else:
            ori_h, ori_w = im.shape[:2]
            im = cv2.resize(im, (w, h))
            labels = np.array(labels, dtype=np.float64)
            labels[:, [0, 2]] *= (w / ori_w)
            labels[:, [1, 3]] *= (h / ori_h)
            labels = np.array(labels, dtype=np.int64)

            img = (im * r + im2 * (1 - r)).astype(np.uint8)
            labels2.extend(labels.tolist())
            return img, labels2
    else:
        im = cv2.resize(im, (w, h))
        img = (im * r + im2 * (1 - r)).astype(np.uint8)
        if label_type == "101":
            return img, labels
        else:
            labels.extend(labels2)
            labels = list(set(labels))
            return img, labels


func_dict = {
    "flip": flip,
    "rotate": rotate,
    "cut": cut,
    "affine": affine,
    "mosaic": get_mosaic_data,
    "scale": scale,
    "mix": mix,
    "noise": noise,
    "wipe": wipe,
    "color": color,
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation
}
