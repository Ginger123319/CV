import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

transform = transforms.Compose([transforms.RandomErasing()])


# 等比例缩放为固定尺寸(正方形)，不足的地方补黑条
def pic_resize(img_data, target_len):
    img_data = np.array(img_data)
    h = img_data.shape[0]
    w = img_data.shape[1]
    max_len = max(h, w)
    # 计算缩放比例
    ratio = target_len / max_len
    h = int(h * ratio)
    w = int(w * ratio)
    # print(h, w)
    # 按比例缩放
    dst = cv2.resize(img_data, (w, h))
    # 给缩放后的图片加黑边，在下面或者右边添加
    # 不需要分类，有一条边为416，计算结果就为0
    dst = cv2.copyMakeBorder(dst, 0, 416 - h, 0, 416 - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    dst = Image.fromarray(dst)
    # 返回处理后的图片数据以及缩放比例
    return dst, ratio


def show_compare_img(original_img, transform_img, option):
    _, axes = plt.subplots(1, 3)
    # 显示图像
    axes[0].imshow(original_img[0])
    axes[1].imshow(original_img[1])
    axes[2].imshow(transform_img)
    # 设置子标题
    axes[0].set_title("original image1")
    axes[1].set_title("original image2")
    axes[2].set_title("{} image".format(option))
    plt.show()


def mix_pic(images, alpha):
    img1, img2 = images
    blend_img = Image.blend(img1, img2, alpha)
    return blend_img


if __name__ == '__main__':
    image_datas = []
    # 图片路径
    root = r"D:\Python\test_jzyj\augment\images"
    # 遍历图片
    _, _, image_names = next(os.walk(root))
    # 遍历所有图片
    for name in image_names:
        # 通道为BGR，形状为HWC的一个ndarray
        img = Image.open(os.path.join(root, name))
        img = pic_resize(img, 416)[0]
        image_datas.append(img)
    mix_img = mix_pic(image_datas[0:2], 0.5)
    show_compare_img(image_datas[0:2], mix_img, "mixup")
