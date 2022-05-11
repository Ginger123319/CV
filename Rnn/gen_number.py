from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os


# 随机数字
def ranNum():
    a = str(random.randint(0, 9))
    a = chr(random.randint(48, 57))
    b = chr(random.randint(65, 90))  # 大写字母
    c = chr(random.randint(97, 122))  # 小写字母
    d = ord(a)
    return a


# 随机颜色1
def ranColor1():
    return (random.randint(65, 255),
            random.randint(65, 255),
            random.randint(65, 255))


# 随机颜色2
def ranColor2():
    return (random.randint(32, 127),
            random.randint(32, 127),
            random.randint(32, 127))


# 240*60
w = 240
h = 60

font = ImageFont.truetype("arial.ttf", 40)
for i in range(1):
    image = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=ranColor1())

    filename = ""
    for j in range(4):
        ch = ranNum()
        filename += ch
        draw.text((60 * j + 10, 10), (ch), font=font, fill=ranColor2())

    # 模糊:
    image = image.filter(ImageFilter.BLUR)
    # image.show()

    if not os.path.exists("../../source/code"):
        os.makedirs("../../source/code")
    image_path = r"../../source/code"
    image.save("{0}/{1}.jpg".format(image_path, filename))
    print(i)
