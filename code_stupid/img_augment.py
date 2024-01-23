import math
import os
import random
import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
from matplotlib import pyplot as plt
from xml.etree import ElementTree as ET


# 生成随机数
def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def show_compare_img(original_img, transform_img, option):
    _, axes = plt.subplots(1, 2)
    # 显示图像
    axes[0].imshow(original_img)
    axes[1].imshow(transform_img)
    # 设置子标题
    axes[0].set_title("original image")
    axes[1].set_title("{} image".format(option))
    plt.show()


# 随机进行上下左右翻转
def flip(input_img):
    img = input_img[0]
    flag = random.randint(-1, 1)
    img = cv2.flip(img, flag)
    return img


# 围绕中心点，任意角度进行旋转
def rotate_img_point(input_img, theta=45):
    """
    :param cx:中心点x坐标
    :param cy:中心点y坐标
    :param theta: 旋转角度
    :param input_img: 输入图片
    :return: 返回图片和标签坐标信息
    """
    img = input_img[0]
    cy = 0.5 * img.shape[0]
    cx = 0.5 * img.shape[1]
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1)
    # 定义旋转后图片的宽和高
    height, width = img.shape[:2]
    # 围绕给定的点逆时针旋转\theta度
    rotate_img = cv2.warpAffine(img, M, (width, height))
    return rotate_img


# 按照给定的左上角和右下角坐标进行裁剪
def crop(input_img, x1=0.3, y1=0.3, x2=0.8, y2=0.8):
    h, w = input_img[0].shape[:2]
    crop_x1 = int(x1 * w)
    crop_y1 = int(y1 * h)
    crop_x2 = int(x2 * w)
    crop_y2 = int(y2 * h)
    img = np.copy(input_img[0])
    if crop_x1 < crop_x2 <= w and crop_y1 < crop_y2 <= h:
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        return crop_img
    else:
        return img


# 随机改变颜色
def turn_color(image, rgain=0.5, ggain=0.5, bgain=0.5):
    img = image[0]
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
    else:
        return img


# 增强亮度
def brighter(img, vgain=0.4):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param vgain:       HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    image = img[0]
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
        return image_dst
    else:
        return image


# 增强饱和度
def saturation(img, sgain=0.7):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param sgain:       HSV 中的 s 扰动系数，yolov5：0.7
    :return:
    """
    image = img[0]
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
        return image_dst
    else:
        return image


# HSV变换
def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """
    HSV color-space augmentation
    :param img:       待增强的图片
    :param vgain:       HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    image = img[0]
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
        return image_dst
    else:
        return image


# 等比缩放
def scale_img(img, scale=0.5, is_Equal=True):
    input_img = img[0]
    assert 0.5 <= scale <= 2
    if is_Equal:
        # 定义宽缩放的倍数
        fx = scale
        # 定义高缩放的倍数
        fy = scale
    else:
        fx, fy = np.random.uniform(0.5, 1.5, 2)
    # 定义一个图像缩放矩阵
    M = np.array([[fx, 0, 0], [0, fy, 0]])
    # 获取图片的宽和高
    height, width = input_img.shape[:2]
    # 定义缩放后图片的大小
    scale_img = cv2.warpAffine(input_img, M, (int(width * fx), int(height * fy)))

    return scale_img


def random_noise(img, noise_num=0.05):
    """
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param img: 需要加噪的图片
    :param noise_num: 添加的噪音点比例，【0.05~0.1之间】
    :return: img_noise
    """
    img_noise = np.copy(img[0])
    if 0.1 >= noise_num > 0:
        rows, cols, chn = img_noise.shape
        noise_num = int(noise_num * rows * cols)
        # 加噪声
        for i in range(noise_num):
            x = np.random.randint(0, rows)  # 随机生成指定范围的整数
            y = np.random.randint(0, cols)
            if i % 2 == 0:
                img_noise[x, y, :] = 0
            else:
                img_noise[x, y, :] = 255
    return img_noise


def random_erase(image, p=1, sl=0.1, sh=0.4, r1=0.3):
    """Random erasing the an rectangle region in Image.
       Args:
           sl: min erasing area region
           sh: max erasing area region
           r1: min aspect ratio range of earsing region
           p: probability of performing random erasing
           image: opencv numpy array in form of [w, h, c] range
                     from [0, 255]
       Returns:
           erased img
    """
    img = np.copy(image[0])
    p = p
    s = (sl, sh)
    r = (r1, 1 / r1)
    assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

    if random.random() > p:
        return img
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
                return img


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


# XY轴错切
def shearXY(image, shear=10):
    img = image[0]
    height = img.shape[0]
    width = img.shape[1]
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    M = S[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return affine_img


# XY轴平移
def translateXY(image, translate=.1):
    img = image[0]
    height = img.shape[0]
    width = img.shape[1]
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1]  # y translation (pixels)
    M = T[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return affine_img


# 仿射变换
def random_affine(image, degrees=10, translate=.1, scale=.1, shear=20, border=0):
    img = image[0]
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
    M = S[:2]
    # 进行仿射变化
    affine_img = cv2.warpAffine(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return affine_img


# 增强对比度
def contrast(image, v=1.5):
    img = image[0]
    img = Image.fromarray(img)
    enh_con = ImageEnhance.Contrast(img)
    img_contrasted = enh_con.enhance(v)
    img_contrasted = np.array(img_contrasted)
    return img_contrasted


def mixup(img):
    # img为包含两张图片的一个列表
    im = img[0][0]
    im2 = img[1][0]
    if im.shape == im2.shape:
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
    else:
        print("image size diff!")
    return im


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
    img = image_list[4]
    # -----------------------------------------
    # 混合两张图片
    # ------------------------------------------
    mix_img = mixup(image_list[:2])
    show_compare_img(image_list[:2][0][0], mix_img, "mix")
    # -----------------------------------------
    # 随机进行翻转
    # ------------------------------------------
    flip_img = flip(img)
    show_compare_img(img[0], flip_img, "flip")
    # -----------------------------------------
    # 指定角度和中心点进行旋转
    # ------------------------------------------
    rotate_img = rotate_img_point(input_img=img)
    show_compare_img(img[0], rotate_img, "rotate")

    # -----------------------------------------
    # 随机进行裁剪
    # ------------------------------------------
    crop_img = crop(img)
    show_compare_img(img[0], crop_img, "crop")

    # -----------------------------------------
    # 对输入进来的图片进行缩放
    # ------------------------------------------
    scale = rand(scale_low, scale_high)
    scale_img = scale_img(img, scale=1.5, is_Equal=True)
    show_compare_img(img[0], scale_img, "scale")
    # -----------------------------------------
    # 对输入进来的图片进行随机色彩变换
    # ------------------------------------------
    # r, g, b = [rand() for i in range(3)]
    color_img = turn_color(img)
    show_compare_img(img[0], color_img, "color_rgb")
    # -----------------------------------------
    # 对输入进来的图片进行随机的HSV变换
    # ------------------------------------------
    v_img = brighter(img)
    show_compare_img(img[0], v_img, "brighter")
    s_img = saturation(img)
    show_compare_img(img[0], s_img, "saturation")
    # hsv_img = augment_hsv(img)
    # show_compare_img(img[0], hsv_img, "hsv")
    # -----------------------------------------
    # 对输入进来的图片随机添加椒盐噪声
    # np.copy深拷贝，图片存储在另一个地址
    # ------------------------------------------
    noise_img = random_noise(img)
    show_compare_img(img[0], noise_img, "noise")
    # -----------------------------------------
    # 对输入进来的图片随机擦除
    # ------------------------------------------
    erase_img = random_erase(img)
    show_compare_img(img[0], erase_img, "erase")
    # -----------------------------------------
    # 对输入进来的图片进行仿射变换
    # ------------------------------------------
    # affine_img = random_affine(img)
    # show_compare_img(img[0], affine_img, "affine")
    # -----------------------------------------
    # 全局增强图片对比度
    # ------------------------------------------
    contrast_img = contrast(img)
    show_compare_img(img[0], contrast_img, "contrast")
    # -----------------------------------------
    # 仿射变换-错切
    # ------------------------------------------
    shear_img = shearXY(img, shear=15)
    show_compare_img(img[0], shear_img, "shear")
    # -----------------------------------------
    # 仿射变换-平移
    # ------------------------------------------
    translate_img = translateXY(img, translate=.2)
    show_compare_img(img[0], translate_img, "translate")
