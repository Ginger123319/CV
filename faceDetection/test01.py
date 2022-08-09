import torch
import os
from PIL import Image, ImageDraw

path = r"D:\Python\source\FACE\celebA\img\img_celeba\img_celeba"
tag_path = r"D:\Python\source\FACE\celebA\Anno\list_bbox_celeba.txt"
counter = 1
with open(tag_path, "r") as a_file:
    a_file.readline()
    a_file.readline()
    for line in a_file:
        line = line.split()
        filename = line[0]
        li = list(map(int, line[1:]))
        print(li)
        li[2] += li[0]
        li[3] += li[1]
        print(li)
        print(filename)
        filename = os.path.join(path, filename)
        img = Image.open(filename)
        w, h = img.size
        print(w, h)
        li[0] = li[0] * 300 / w
        li[1] = li[1] * 300 / h
        li[2] = li[2] * 300 / w
        li[3] = li[3] * 300 / h
        print(li)
        # # img.show()
        # draw = ImageDraw.Draw(img)
        # draw.rectangle(li, outline="pink", width=3)
        # img.show()
        img = img.resize((300, 300))
        draw = ImageDraw.Draw(img)
        draw.rectangle(li, outline="pink", width=3)
        img.show()
        # print(filename)

        if counter >= 4:
            break
        counter += 1
