import math
import os
import random
import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
from matplotlib import pyplot as plt
from xml.etree import ElementTree as ET
from mosaic_all import get_mosaic_data


# 生成随机数
def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


# 绘制单张原图和增强后的图片（根据label_Box标签确认是否需要画出目标框）
def draw_rect(original_img, transform_img, target, option, label_box=True):
    if label_box:
        for box in target:
            x1, y1, x2, y2 = box
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                cv2.rectangle(transform_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            else:
                continue
    _, axes = plt.subplots(1, 2)
    # 显示图像
    axes[0].imshow(original_img)
    axes[1].imshow(transform_img)
    # 设置子标题
    axes[0].set_title("original image")
    axes[1].set_title("{} image".format(option))
    plt.show()


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
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    # clip
    new[:, [0, 2]] = new[:, [0, 2]].clip(2, width - 2)
    new[:, [1, 3]] = new[:, [1, 3]].clip(2, height - 2)
    return new.astype(np.uint)


# 随机进行上下左右翻转
def flip(input_img, label_box=True):
    img = input_img[0]
    target = np.array(input_img[1])
    h, w = img.shape[:2]
    flag = random.randint(-1, 1)
    img = cv2.flip(img, flag)
    # 如果标签是框，执行框的坐标转换代码
    if label_box:
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
    return img, target


# 按照给定的左上角和右下角坐标进行裁剪
def crop(input_img, lu_max=0.3, rb_min=0.8, label_box=True):
    """
    HSV color-space augmentation
    :param lu_max 左上角坐标的最大比例
    :param rb_min 右下角坐标的最小比例
    :return:
    """
    # lu_max 左上角坐标的最大比例
    # rb_min 右下角坐标的最小比例
    assert 0.1 < lu_max <= 0.4
    assert 0.8 <= rb_min < 1
    crop_x1y1 = np.random.uniform(0.1, lu_max, 2)
    crop_x2y2 = np.random.uniform(rb_min, 1, 2)
    x1, y1 = crop_x1y1
    x2, y2 = crop_x2y2
    h, w = input_img[0].shape[:2]
    crop_x1 = int(x1 * w)
    crop_y1 = int(y1 * h)
    crop_x2 = int(x2 * w)
    crop_y2 = int(y2 * h)
    img = np.copy(input_img[0])
    target = np.array(input_img[1])
    if crop_x1 < crop_x2 <= w and crop_y1 < crop_y2 <= h:
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if label_box:
            # 先做坐标转换
            target[:, [0, 2]] = target[:, [0, 2]] - crop_x1
            target[:, [1, 3]] = target[:, [1, 3]] - crop_y1
            target[:, [1, 3]] = np.clip(target[:, [1, 3]], 2, crop_y2 - crop_y1 - 2)
            target[:, [0, 2]] = np.clip(target[:, [0, 2]], 2, crop_x2 - crop_x1 - 2)
        return crop_img, target
    else:
        return img, target


# 随机改变颜色
def turn_color(image, rgain=0.5, ggain=0.5, bgain=0.5):
    img = image[0]
    target = np.array(image[1])
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
        return img_rgb, target
    else:
        return img, target


# 增强亮度
def brighter(img, vgain=0.4):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param vgain:       HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    image = img[0]
    target = np.array(img[1])
    if vgain:
        # r:[1+0.5gain,1+gain](变得更亮并且更加饱和)
        r = np.random.uniform(0.5, 1, 1) * [vgain] + 1  # random gains
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # cv2.split：通道拆分
        # h:[0~180]色彩, s:[0~255]饱和度, v:[0~255]亮度
        hue, sat, val = cv2.split(image_hsv)
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_val = np.clip(x * r[0], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        val = cv2.LUT(val, lut_val)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)
        # 将hsv格式转为BGR格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image_dst, target
    else:
        return image, target


# 增强饱和度
def saturation(img, sgain=0.7):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param sgain:       HSV 中的 s 扰动系数，yolov5：0.7
    :return:
    """
    image = img[0]
    target = np.array(img[1])
    if sgain:
        # r:[1+0.5gain,1+gain](变得更亮并且更加饱和)
        r = np.random.uniform(0.5, 1, 1) * [sgain] + 1  # random gains
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # cv2.split：通道拆分
        # h:[0~180]色彩, s:[0~255]饱和度, v:[0~255]亮度
        hue, sat, val = cv2.split(image_hsv)
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_sat = np.clip(x * r[0], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        sat = cv2.LUT(sat, lut_sat)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)

        # 将hsv格式转为BGR格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image_dst, target
    else:
        return image, target


# HSV变换
def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param vgain:     HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    image = img[0]
    target = np.array(img[1])
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以 hsv 三通道扰动系数
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # 可以变暗也可以变得更加不饱和
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # cv2.split：通道拆分
        # h:[0~180]色彩, s:[0~255]饱和度, v:[0~255]亮度
        hue, sat, val = cv2.split(image_hsv)
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[0], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)

        # 将hsv格式转为BGR格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return image_dst, target
    else:
        return image, target


# 由于for循环太耗时，改为张量运算的方式进行优化
# def random_noise(img, noise_num=0.01):
#     """
#     添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
#     :param img: 需要加噪的图片
#     :param noise_num: 添加的噪音点比例，【0.05~0.1之间】
#     :return: img_noise
#     """
#     assert 0 < noise_num <= 0.03
#     noise_num = random.uniform(0.03 - noise_num, 0.03 + noise_num)
#     img_noise = np.copy(img[0])
#     target = np.array(img[1])
#     rows, cols, chn = img_noise.shape
#     noise_num = int(noise_num * rows * cols)
#     # 加噪声
#     for i in range(noise_num):
#         x = np.random.randint(0, rows)  # 随机生成指定范围的整数
#         y = np.random.randint(0, cols)
#         if i % 2 == 0:
#             img_noise[x, y, :] = 0
#         else:
#             img_noise[x, y, :] = 255
#     return img_noise, target
def gaussian_noise(img, mean=0.1, sigma=0.1):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.copy(img[0])
    target = np.array(img[1])
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


def salt_pepper_noise(img, noise_num=0.02):
    """
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param img: 需要加噪的图片以及其标签，形如[[img:np.ndarray],[[..],[...]]]
    :param noise_num: 添加的噪音点比例，【0.02~0.05之间】
    :return: img_noise
    """
    img_noise = np.copy(img[0])
    target = np.array(img[1])
    img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 255)
    img_noise = add_uniform_noise(img_noise, noise_num * 0.5, 0)
    return img_noise, target


def random_erase(image, p=1, sl=0.1, sh=0.4, r1=0.3, label_box=True):
    """Random erasing the an rectangle region in Image.
       Args:
           sl: min erasing area region
           sh: max erasing area region
           r1: min aspect ratio range of earsing region
           p: probability of performing random erasing
       """
    """
            perform random erasing
            Args:
                img: opencv numpy array in form of [w, h, c] range
                     from [0, 255]

            Returns:
                erased img
           """
    img = np.copy(image[0])
    target = np.array(image[1])
    p = p
    s = (sl, sh)
    r = (r1, 1 / r1)
    assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

    if random.random() > p:
        return img, target

    else:
        if label_box:
            #
            for box in target:
                x1, y1, x2, y2 = box
                i = 0
                while i < 3000:
                    # Se = random.uniform(*s) * img.shape[0] * img.shape[1]
                    Se = random.uniform(*s) * (y2 - y1) * (x2 - x1)
                    re = random.uniform(*r)

                    He = int(round(math.sqrt(Se * re)))
                    We = int(round(math.sqrt(Se / re)))

                    xe = random.randint(0, img.shape[1])
                    ye = random.randint(0, img.shape[0])
                    i += 1
                    # if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    if x2 >= xe + We > xe >= x1 and y2 >= ye + He > ye >= y1:
                        img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255,
                                                                             size=(He, We, img.shape[2]))
                        break
            return img, target
        else:
            i = 0
            while i < 3000:
                Se = random.uniform(*s) * img.shape[0] * img.shape[1]
                re = random.uniform(*r)

                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))

                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])
                i += 1
                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))
                    return img, target


# 等比例缩放为固定尺寸(正方形)，不足的地方补黑边
def pic_resize(img_data, target_len):
    h = img_data.shape[0]
    w = img_data.shape[1]
    max_len = max(h, w)
    # 计算缩放比例
    ratio = target_len / max_len
    h = int(h * ratio)
    w = int(w * ratio)
    # 按比例缩放
    dst = cv2.resize(img_data, (w, h))
    # 给缩放后的图片加黑边，在下面或者右边添加
    # 不需要分类，有一条边为416，计算结果就为0
    dst = cv2.copyMakeBorder(dst, 0, 416 - h, 0, 416 - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # 返回处理后的图片数据以及缩放比例
    return dst, ratio


# 增强对比度
def contrast(image, v=1.5):
    img = image[0]
    img = Image.fromarray(img)
    target = np.array(image[1])
    enh_con = ImageEnhance.Contrast(img)
    img_contrasted = enh_con.enhance(v)
    img_contrasted = np.array(img_contrasted)
    return img_contrasted, target


# 混合两张图片
def mixup(img):
    # img为包含两张图片的一个列表
    im = img[0][0]
    im2 = img[1][0]
    labels = np.array(img[0][1])
    labels2 = np.array(img[1][1])
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    if im.shape == im2.shape:
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
    else:
        print("image size diff!")
    return im, labels


# 以下均为仿射变换的实现
# 围绕中心点，任意角度进行旋转
def rotate_img_point(input_img, degrees=45, label_box=True):
    """
    :param cx:中心点x坐标
    :param cy:中心点y坐标
    :param theta: 旋转角度
    :param input_img: 输入图片
    :return: 返回图片和标签坐标信息
    """
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    img = input_img[0]
    target = np.array(input_img[1])
    cy = 0.5 * img.shape[0]
    cx = 0.5 * img.shape[1]
    M = cv2.getRotationMatrix2D((cx, cy), a, 1)
    # 定义旋转后图片的宽和高
    height, width = img.shape[:2]
    # 围绕给定的点逆时针旋转\theta度
    rotate_img = cv2.warpAffine(img, M, (width, height))
    if label_box:
        # 标签框坐标转换
        target = transform_affine_labels(target, M, rotate_img)
    return rotate_img, target


# 等比缩放
def scale_img(img, scale=0.5, is_Equal=True, label_box=True):
    input_img = img[0]
    target = np.array(img[1])
    assert 0.5 <= scale <= 1
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
    # 定义缩放后图片的大小
    scale_img = cv2.warpAffine(input_img, M, (int(width * fx), int(height * fy)))
    if label_box:
        # 修改标签
        target = transform_affine_labels(target, M, scale_img)
    return scale_img, target


def shearXY(image, shear=10, label_box=True):
    img = image[0]
    target = np.array(image[1])
    height = img.shape[0]
    width = img.shape[1]
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    M = S[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if label_box:
        # 进行标签转换
        target = transform_affine_labels(target, M, affine_img)
    return affine_img, target


def translateXY(image, translate=.1, label_box=True):
    img = image[0]
    target = np.array(image[1])
    height = img.shape[0]
    width = img.shape[1]
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1]  # y translation (pixels)
    M = T[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if label_box:
        # 进行标签转换
        target = transform_affine_labels(target, M, affine_img)
    return affine_img, target


# 仿射变换
def random_affine(image, degrees=10, translate=.1, scale=.1, shear=10, border=0, label_box=True):
    img = image[0]
    target = np.array(image[1])
    # 最终输出的图像尺寸，等于img4.shape / 2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

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
    if label_box:
        # 进行标签转换
        target = transform_affine_labels(target, M, affine_img)
    return affine_img, target


if __name__ == '__main__':
    # 设置偏移量
    min_offset_x = 0.3
    min_offset_y = 0.3
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.3
    # 图片路径
    img_root = r"D:\Python\test_jzyj\augment\images"
    xml_root = r"D:\Python\test_jzyj\augment\labels"
    image_list = []  # 存放每张图像和该图像对应的检测框坐标信息
    for xml in os.listdir(xml_root):
        image_box = []
        img_path = os.path.join(img_root, xml.split(".")[0] + ".jpg")
        xml_path = os.path.join(xml_root, xml)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 读取检测框信息
            with open(xml_path, 'r') as new_f:
                #  getroot()获取根节点
                root = ET.parse(xml_path).getroot()

            # findall查询根节点下的所有直系子节点，find查询根节点下的第一个直系子节点
            for obj in root.findall('object'):
                obj_name = obj.find('name').text  # 目标名称
                bndbox = obj.find('bndbox')
                left = eval(bndbox.find('xmin').text)  # 左上坐标x
                top = eval(bndbox.find('ymin').text)  # 左上坐标y
                right = eval(bndbox.find('xmax').text)  # 右下坐标x
                bottom = eval(bndbox.find('ymax').text)  # 右下坐标y

                # 保存每张图片的检测框信息
                image_box.append([left, top, right, bottom])  # [[x1,y1,x2,y2],[..],[..]]
            image_list.append([img, image_box])
        except Exception as e:
            print(e)
            continue
    # -----------------------------------------
    # image_list是一个列表；每个元素是一个列表；每个元素的0号索引位置是图片
    # 后面的索引位置是框（也是一个列表，元素也是列表（代表每一个框的信息））
    # ------------------------------------------
    img = image_list[0]
    if len(img[1]) > 0:
        # 用于控制使用目标检测数据增强还是图像分类数据增强
        is_detect = True
        # -----------------------------------------
        # 对输入进来的图片进行仿射变换
        # ------------------------------------------
        affine_img, boxes = random_affine(img, label_box=is_detect)
        # draw_rect(img[0], affine_img, boxes, "affine", label_box=is_detect)

        # -----------------------------------------
        # 随机进行翻转
        # ------------------------------------------
        flip_img, boxes = flip(img, label_box=is_detect)
        # draw_rect(img[0], flip_img, boxes, "flip", label_box=is_detect)
        # -----------------------------------------
        # 指定角度和中心点进行旋转
        # ------------------------------------------
        rotate_img, boxes = rotate_img_point(input_img=img, label_box=is_detect)
        # draw_rect(img[0], rotate_img, boxes, "rotate", label_box=is_detect)

        # -----------------------------------------
        # 随机进行裁剪
        # ------------------------------------------
        crop_img, boxes = crop(img, lu_max=0.3, rb_min=0.8, label_box=is_detect)
        # draw_rect(img[0], crop_img, boxes, "crop", label_box=is_detect)
        # -----------------------------------------
        # 对输入进来的图片进行缩放
        # ------------------------------------------
        scale_img, boxes = scale_img(img, scale=0.5, is_Equal=False, label_box=is_detect)
        # draw_rect(img[0], scale_img, boxes, "scale", label_box=is_detect)
        # -----------------------------------------
        # 对输入进来的图片随机擦除
        # ------------------------------------------
        erase_img, boxes = random_erase(img, label_box=is_detect)
        # draw_rect(img[0], erase_img, boxes, "erase", label_box=is_detect)
        # -----------------------------------------
        # 仿射变换-错切
        # ------------------------------------------
        shear_img, boxes = shearXY(img, shear=15, label_box=is_detect)
        # draw_rect(img[0], shear_img, boxes, "shear", label_box=is_detect)
        # -----------------------------------------
        # 仿射变换-平移
        # ------------------------------------------
        translate_img, boxes = translateXY(img, translate=.2, label_box=is_detect)
        # draw_rect(img[0], translate_img, boxes, "translate", label_box=is_detect)
        # ----------------------------------------------
        # 马赛克数据增强，通过label_box控制是图像分类还是目标检测的马赛克操作
        # -------------------------------------------------
        mosaic_img, boxes = get_mosaic_data(image_list[:4], input_shape=[416, 416], label_box=is_detect)
        # draw_rect(image_list[0][0], mosaic_img, boxes, "mosaic", label_box=is_detect)
        # --------------------------------------------------------------------------------------------------------------
        #                               不需要做标签转换的，直接返回输入的标签即可
        # --------------------------------------------------------------------------------------------------------------
        # 对输入进来的图片进行随机色彩变换
        # ------------------------------------------
        # r, g, b = [rand() for i in range(3)]
        color_img, boxes = turn_color(img)
        # draw_rect(img[0], color_img, boxes, "color_rgb", label_box=is_detect)
        # -----------------------------------------
        # 混合两张图片
        # ------------------------------------------
        mix_img, boxes = mixup(image_list[:2])
        # draw_rect(image_list[:2][0][0], mix_img, boxes, "mix", label_box=is_detect)
        # -----------------------------------------
        # 对输入进来的图片进行随机的HSV变换（亮度和饱和度）
        # ------------------------------------------
        v_img, boxes = brighter(img)
        # draw_rect(img[0], v_img, boxes, "brighter", label_box=is_detect)
        s_img, boxes = saturation(img)
        # draw_rect(img[0], s_img, boxes, "saturation", label_box=is_detect)
        # -----------------------------------------
        # 对输入进来的图片随机添加椒盐噪声
        # np.copy深拷贝，图片存储在另一个地址
        # ------------------------------------------
        noise_img, boxes = salt_pepper_noise(img)
        draw_rect(img[0], noise_img, boxes, "noise", label_box=is_detect)
        # -----------------------------------------
        # 对输入进来的图片随机添加高斯噪声
        # ------------------------------------------
        gaussian_img, boxes = gaussian_noise(img)
        draw_rect(img[0], gaussian_img, boxes, 'gaussian', label_box=is_detect)
        # -----------------------------------------
        # 全局增强图片对比度
        # ------------------------------------------
        contrast_img, boxes = contrast(img)
        # draw_rect(img[0], contrast_img, boxes, "contrast", label_box=is_detect)
