import PIL.ImageFont as imgfont
import PIL.Image as image
import PIL.ImageDraw as draw
import random

font = imgfont.truetype("font.TTF", 60)

w = 240
h = 120


def img():
    return image.new("RGB", (w, h), (255, 255, 255))


# 生成一块空白的画板


def b_color():
    return (random.randint(64, 255),
            random.randint(64, 255),
            random.randint(64, 255))


# 生成随机背景色 print(b_color())


def rand_char():
    return chr(random.randint(65, 90))


# 生成随机字符 print(rand_char())


def f_color():
    return (random.randint(32, 128),
            random.randint(32, 128),
            random.randint(32, 128))


# 前景色生成


# img().show()
# 实现思路，生成一个白板，在白板上的每一个点上背景色，然后在写字母，最后把字母的颜色确认也就是前景色
# 使用到了PIL的ImageFont、Image、ImageDraw三个库

if __name__ == '__main__':
    img = img()
    # 开始在空白画板img上进行绘制
    image = draw.Draw(img)
    for x in range(w):
        for y in range(h):
            # 画点
            image.point((x, y), b_color())
    for i in range(4):
        # 写字
        image.text((60 * i + 10, 30), text=rand_char(), fill=f_color(), font=font)
    img.show()
