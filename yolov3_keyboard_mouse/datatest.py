import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFont
import cfg
import cv2
import PIL.ImageDraw as draw

name = {
    0: "鼠标",
    1: "键盘"
}
color = {
    0: "red",
    1: "black"
}
with open(cfg.train_path) as f:
    for line in f.readlines():
        print(line.split()[0])
        img_path = os.path.join(cfg.img_path, line.split()[0])
        print(img_path)
        im = Image.open(img_path)
        # im.show()
        print(len(line.split()[1:]))
        for i in range(len(line.split()[1:]) // 5):
            i = i*5+1
            cls, x1, y1, x2, y2 = list(map(int, line.split()[i:i + 5]))  # 将自信度和坐标及类别分别解包出来
            print(cls, x1, y1, x2, y2)
            x3 = x1 - x2 / 2
            y3 = y1 - y2 / 2
            x4 = x1 + x2 / 2
            y4 = y1 + y2 / 2
            print(x3, y3, x4, y4)
            # print(int(cls.item()))
            # print(round(c.item(),4))#取值并保留小数点后4位
            img_draw = draw.ImageDraw(im)
            font = ImageFont.truetype("simsun.ttc", 25, encoding="unic")
            img_draw.rectangle((x3, y3, x4, y4), outline=color[cls], width=4)  # 画框

            img_draw.text((max(x1, 0) + 20, max(y1, 0) + 5), fill=color[cls],
                          text=name[cls], font=font, width=2)
            plt.clf()
            plt.ion()
            plt.axis('off')
            plt.imshow(im)
            # plt.show()
            plt.pause(3)
            plt.close()
